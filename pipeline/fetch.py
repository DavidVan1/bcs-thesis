"""
Automatic data fetching for a PhiSat-2 scene.

Given only the phisat folder (AOCS.json + GL_scene_0.json), downloads:

  1. Sentinel-2 L1C imagery  – Google Earth Engine (COPERNICUS/S2_HARMONIZED)
  2. Copernicus DEM GLO-30    – Google Earth Engine (COPERNICUS/DEM/GLO30)
  3. Copernicus GCP database  – ESA S2 GRI HTTPS tiles

Usage (CLI):
    python -m pipeline.run <scene> fetch
    python -m pipeline.run new_scene fetch --phisat-dir phisat/phisat_new

Usage (API):
    from pipeline.fetch import run_fetch
    run_fetch(config)

Google Earth Engine authentication:
    Run once before first use:
        earthengine authenticate
    or inside the environment:
        python -c "import ee; ee.Authenticate()"
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import SceneConfig, PROJECT_ROOT


# ═══════════════════════════════════════════════════════════════════════════
# Footprint helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_footprint(phisat_dir: Path) -> Tuple[float, float, float, float]:
    """
    Parse GL_scene_0.json and return the tight bounding box
    (lon_min, lat_min, lon_max, lat_max) of the scene.
    """
    gl_path = phisat_dir / "geolocation" / "GL_scene_0.json"
    if not gl_path.exists():
        raise FileNotFoundError(
            f"Geolocation file not found: {gl_path}\n"
            "Make sure the PhiSat folder contains geolocation/GL_scene_0.json"
        )

    with open(gl_path) as f:
        data = json.load(f)

    pts = data.get("Geolocated_Points", [])
    if not pts:
        raise ValueError(f"No geolocated points in {gl_path}")

    lons = [p["Lon"] for p in pts]
    lats = [p["Lat"] for p in pts]

    return min(lons), min(lats), max(lons), max(lats)


def _expand_bbox(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    margin_deg: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Add a margin around the bounding box."""
    return (
        lon_min - margin_deg,
        lat_min - margin_deg,
        lon_max + margin_deg,
        lat_max + margin_deg,
    )


def _load_acquisition_time(phisat_dir: Path) -> Optional[str]:
    """
    Return the UTC acquisition date (YYYY-MM-DD) from AOCS.json.
    ADCSTimeSec is a plain Unix timestamp (seconds since 1970-01-01 UTC).
    """
    aocs_path = phisat_dir / "AOCS.json"
    if not aocs_path.exists():
        return None

    with open(aocs_path) as f:
        data = json.load(f)

    try:
        unix_sec = int(data["Acquisitions"][0]["ADCSTimeSec"])
    except (KeyError, IndexError):
        return None

    import datetime
    dt = datetime.datetime.utcfromtimestamp(unix_sec)
    return dt.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════
# Sentinel-2 download via Google Earth Engine
# ═══════════════════════════════════════════════════════════════════════════

def _make_tci(image):
    """
    Convert a Sentinel-2 L1C image (DN 0-10000) to a TCI-style 3-band uint8
    RGB image matching the official ESA TCI product (R=B4, G=B3, B=B2).
    ESA TCI formula: clamp(DN / 3558 * 255, 0, 255).toByte()
    (3558 is the hardcoded divisor used by the ESA L1C processor)

    Note: copyProperties() returns ee.Element (EE API quirk), so we cast
    back to ee.Image via ee.Image() to preserve the .clip() method.
    """
    import ee
    tci = (
        image.select(["B4", "B3", "B2"])
             .divide(3558).multiply(255)
             .clamp(0, 255)
             .toByte()
             .rename(["R", "G", "B"])
    )
    # Cast back to ee.Image (copyProperties returns ee.Element)
    return ee.Image(tci)


def download_sentinel(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    acq_date: Optional[str],
    output_dir: Path,
    *,
    date_window_days: int = 90,
    max_cloud_pct: float = 5.0,
    scale_m: float = 10.0,
) -> Path:
    """
    Download the least-cloudy Sentinel-2 TCI (True Colour Image) covering
    the footprint from Google Earth Engine.

    Output: 3-band uint8 GeoTIFF (R=B4, G=B3, B=B2) equivalent to the
    official ESA TCI product — directly usable by the feature matcher.

    Cloud selection uses S2_SR_HARMONIZED (has SCL band) to compute the
    actual cloud fraction inside the footprint.  The TCI mosaic is then
    built from S2_HARMONIZED (L1C) for the same date.
    """
    try:
        import ee
        from geedim.mask import BaseImage
    except ImportError:
        raise ImportError(
            "Google Earth Engine packages not found.\n"
            "Install with:  pip install earthengine-api geedim\n"
            "Then authenticate once:  earthengine authenticate --auth_mode=notebook"
        )

    try:
        ee.Initialize(opt_url="https://earthengine.googleapis.com")
    except ee.EEException:
        raise RuntimeError(
            "Google Earth Engine not authenticated.\n"
            "Run:  earthengine authenticate --auth_mode=notebook"
        )

    print(f"\n  Sentinel-2 TCI — bounding box: "
          f"[{lon_min:.4f},{lon_max:.4f}] x [{lat_min:.4f},{lat_max:.4f}]")

    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    def _best_date(start: str, end: str, cloud_limit: float,
                   min_coverage: float = 0.90):
        """
        Find the date with the lowest cloud fraction AND sufficient spatial
        coverage of the ROI.

        For each candidate date the per-granule SCL cloud fraction is scored
        *and* the mosaic's valid-pixel coverage over the ROI is checked.
        Dates where coverage < *min_coverage* are discarded, then the date
        with the lowest cloud fraction is returned.

        Uses S2_SR_HARMONIZED (has SCL band) for scoring.
        Returns (date_str, cloud_frac) or (None, None).
        """
        sr_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",
                                  min(cloud_limit * 3, 100)))
        )
        size = sr_col.size().getInfo()
        if size == 0:
            return None, None

        print(f"    {size} granule(s) in window — scoring ROI cloud + coverage ...")

        # ── per-granule cloud scoring ──
        def add_date_cloud(img):
            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            scl = img.select("SCL")
            cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
            stats = cloud_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region, scale=60, maxPixels=1e7, bestEffort=True,
            )
            frac = ee.Number(stats.get("SCL")).multiply(100)
            return img.set("DATE_DAY", date_str).set("CLOUD_ROI", frac)

        scored = sr_col.map(add_date_cloud)
        dates_list = scored.aggregate_array("DATE_DAY").getInfo()
        fracs_list = scored.aggregate_array("CLOUD_ROI").getInfo()
        if not dates_list:
            return None, None

        from collections import defaultdict
        date_fracs: dict = defaultdict(list)
        for d, f in zip(dates_list, fracs_list):
            if f is not None:
                date_fracs[d].append(f)

        # ── rank candidates: lowest cloud first ──
        candidates = sorted(date_fracs.keys(),
                            key=lambda d: sum(date_fracs[d]) / len(date_fracs[d]))

        # ── check spatial coverage for top candidates ──
        for date_str in candidates:
            frac = sum(date_fracs[date_str]) / len(date_fracs[date_str])
            if frac >= cloud_limit:
                break  # remaining are worse — give up

            next_day = (datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            day_mosaic = (
                sr_col.filterDate(date_str, next_day)
                .select("B4").mosaic()       # any band will do
            )
            # fraction of ROI pixels that have valid (non-null) data
            valid_mask = day_mosaic.mask()    # 1 where data, 0 where masked
            cov_stat = valid_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region, scale=60, maxPixels=1e7, bestEffort=True,
            )
            coverage = ee.Number(cov_stat.get("B4")).getInfo()
            if coverage is None:
                coverage = 0.0
            n_gran = len(date_fracs[date_str])
            print(f"    {date_str}  cloud {frac:5.2f}%  coverage {coverage*100:5.1f}%  "
                  f"({n_gran} granule(s))")

            if coverage >= min_coverage:
                return date_str, frac

        print(f"    No date with cloud < {cloud_limit}% and coverage >= "
              f"{min_coverage*100:.0f}%")
        return None, None

    # ── pass 1: around acquisition date ───────────────────────────
    import datetime
    best_date = cloud_frac = None
    if acq_date:
        dt = datetime.datetime.strptime(acq_date, "%Y-%m-%d")
        td = datetime.timedelta(days=date_window_days)
        start = (dt - td).strftime("%Y-%m-%d")
        end   = (dt + td).strftime("%Y-%m-%d")
        print(f"  Pass 1 — {start} -> {end}  (ROI cloud <= {max_cloud_pct}%)")
        best_date, cloud_frac = _best_date(start, end, max_cloud_pct)

    # ── pass 2: full archive ───────────────────────────────────────
    if best_date is None:
        print(f"  Pass 2 — full archive  (ROI cloud <= {max_cloud_pct}%)")
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", max_cloud_pct)

    # ── pass 3: relaxed cloud ──────────────────────────────────────
    if best_date is None:
        print(f"  Pass 3 — full archive, cloud <= 30%")
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", 30.0)

    # ── pass 4: no cloud filter ────────────────────────────────────
    if best_date is None:
        print(f"  Pass 4 — full archive, no cloud filter")
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", 100.0)
        if best_date is None:
            raise RuntimeError("No Sentinel-2 images found for the given footprint.")

    print(f"  Selected date       : {best_date}")
    print(f"  Cloud (footprint)   : {cloud_frac:.2f}%")

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"sentinel_TCI_{best_date}.tif"

    if out_tif.exists():
        print(f"  Already exists — skipping download: {out_tif.name}")
        return out_tif

    # Build TCI mosaic from L1C (S2_HARMONIZED) on the best date.
    # S2_HARMONIZED starts from 2017; for earlier dates fall back to
    # using the SR collection (B4/B3/B2 are also present there).
    next_day = (datetime.datetime.strptime(best_date, "%Y-%m-%d")
                + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    def _tci_col(collection_id: str) -> "ee.ImageCollection":
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filterDate(best_date, next_day)
        )

    l1c_day = _tci_col("COPERNICUS/S2_HARMONIZED")
    if l1c_day.size().getInfo() == 0:
        print(f"  L1C collection empty for {best_date} — using SR collection for TCI")
        l1c_day = _tci_col("COPERNICUS/S2_SR_HARMONIZED")

    n_granules = l1c_day.size().getInfo()
    print(f"  Mosaicking {n_granules} granule(s) for {best_date}")
    tci_mosaic = _make_tci(l1c_day.mosaic()).clip(region)
    gd_img = BaseImage(tci_mosaic)

    print(f"  Downloading TCI to {out_tif} ...")
    gd_img.download(
        str(out_tif),
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        dtype="uint8",
        overwrite=True,
    )
    print(f"  Saved: {out_tif}")
    return out_tif


# ═══════════════════════════════════════════════════════════════════════════
# DEM download via Google Earth Engine
# ═══════════════════════════════════════════════════════════════════════════

def download_dem(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    output_path: Path,
    *,
    scale_m: float = 30.0,
) -> Path:
    """
    Download Copernicus GLO-30 DEM for the footprint from Google Earth Engine.

    The result is a single GeoTIFF in EPSG:4326 at ~30 m resolution.
    """
    try:
        import ee
        from geedim.mask import BaseImage
    except ImportError:
        raise ImportError(
            "Google Earth Engine packages not found.\n"
            "Install with:  pip install earthengine-api geedim"
        )

    try:
        ee.Initialize(opt_url="https://earthengine.googleapis.com")
    except ee.EEException:
        raise RuntimeError(
            "Google Earth Engine not authenticated.\n"
            "Run:  earthengine authenticate"
        )

    print(f"\n  Copernicus DEM GLO-30 — bbox: "
          f"[{lon_min:.4f},{lon_max:.4f}] × [{lat_min:.4f},{lat_max:.4f}]")

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Already exists — skipping download: {output_path.name}")
        return output_path

    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .filterBounds(region)
        .select("DEM")
        .mosaic()
        .clip(region)
    )

    gd_img = BaseImage(dem)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading to {output_path} …")
    gd_img.download(
        str(output_path),
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        dtype="float32",
        overwrite=True,
    )
    print(f"  ✓ Saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# GCP database download (ESA Sentinel-2 GRI)
# ═══════════════════════════════════════════════════════════════════════════

# S2 GRI base URL  (the official ESA/DLR Sentinel-2 Geometric Reference Image)
_GRI_BASE_URL = (
    "https://s2gri.copernicus.eu/gri-db/download/v3"
)

# Alternative public mirror (no authentication needed)
_GRI_MIRROR = (
    "https://eodata.copernicus.eu/browser/GRI/S2/GRI_DB/v3"
)


def _deg1_tile_names(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
) -> list[str]:
    """
    Return 1°×1° GRI tile names (e.g. 'N37W123') covering the bbox.
    Tiles are named by their south-west corner.
    """
    tiles = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return tiles


def _try_download_url(url: str, dest: Path) -> bool:
    """Attempt to download url → dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except Exception:
        return False


def _wasabi_gri_url(tile: str) -> str:
    """
    Build the public Wasabi S3 URL for a GRI tile tar.gz.

    URL pattern:
      https://s3.eu-central-2.wasabisys.com/s2-mpc/data/S2_GRI/GCP_L1C/
          <NS><DD>/<TILE>_L1C.tar.gz
    where <NS><DD> is the 2-char latitude prefix (e.g. N37, S03).

    Each tar.gz contains:
      <TILE>_L1C/<TILE>.json         — GCP database
      <TILE>_L1C/L1C_chips/*.TIF    — reference chips
    """
    # Extract latitude prefix: first 3 chars of tile name, e.g. "N37" from "N37E013"
    lat_prefix = tile[:3]
    return (
        f"https://s3.eu-central-2.wasabisys.com/s2-mpc/data/S2_GRI"
        f"/GCP_L1C/{lat_prefix}/{tile}_L1C.tar.gz"
    )


def download_gcps(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    gcp_dir: Path,
    *,
    include_chips: bool = True,
) -> list[Path]:
    """
    Download Copernicus S2 GRI GCP database tiles (JSON + L1C chips)
    for the footprint.

    GCP tiles are 1°×1° named by their south-west corner: e.g. N37E013.

    Each tile is downloaded as a single tar.gz from the public Wasabi S3
    bucket — no authentication required:
      https://s3.eu-central-2.wasabisys.com/s2-mpc/data/S2_GRI/GCP_L1C/

    The archive is extracted in-place; only the JSON and (optionally) the
    chips are kept.

    Parameters
    ----------
    gcp_dir : Path
        Destination directory (e.g. gcp/my_scene/).
    include_chips : bool
        Also extract L1C reference chips (required for verify stage).

    Returns
    -------
    List of downloaded JSON paths.
    """
    import tarfile
    import tempfile

    tiles = _deg1_tile_names(lon_min, lat_min, lon_max, lat_max)
    print(f"\n  GCP tiles needed: {tiles}")

    gcp_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = gcp_dir / "L1C_chips"
    if include_chips:
        chip_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []

    for tile in tiles:
        json_dest = gcp_dir / f"{tile}.json"

        # Check if already fully downloaded
        if json_dest.exists():
            print(f"  {tile} — already exists, skipping")
            json_paths.append(json_dest)
            continue

        url = _wasabi_gri_url(tile)
        print(f"  Downloading {tile} from Wasabi S3 …")
        print(f"    {url}")

        # Stream tar.gz into a temp file, then extract
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if not _try_download_url(url, tmp_path):
                print(f"  ✗ Could not download {tile}_L1C.tar.gz")
                print(f"    Manual download: {url}")
                tmp_path.unlink(missing_ok=True)
                continue

            with tarfile.open(tmp_path, "r:gz") as tar:
                # Archive layout: <TILE>_L1C/<TILE>.json
                #                 <TILE>_L1C/L1C_chips/*.TIF
                members_to_extract = []
                for member in tar.getmembers():
                    name = member.name
                    if name.endswith(".json"):
                        # Extract JSON flat into gcp_dir
                        member.name = Path(name).name
                        members_to_extract.append((member, gcp_dir))
                    elif include_chips and "/L1C_chips/" in name and name.endswith(".TIF"):
                        member.name = Path(name).name
                        members_to_extract.append((member, chip_dir))

                for member, dest_dir in members_to_extract:
                    tar.extract(member, path=dest_dir)

            if json_dest.exists():
                n_chips = len(list(chip_dir.glob("*.TIF"))) if include_chips else 0
                print(f"  ✓ {tile}.json extracted"
                      + (f"  ({n_chips} chips)" if include_chips else ""))
                json_paths.append(json_dest)
            else:
                print(f"  ✗ {tile}.json not found inside archive")

        finally:
            tmp_path.unlink(missing_ok=True)

    return json_paths


# ═══════════════════════════════════════════════════════════════════════════
# Scene auto-config builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_scene_config_block(
    scene_name: str,
    phisat_rel: str,
    phisat_image_rel: str,
    metadata_json: Optional[str],
    sentinel_rel: str,
    dem_rel: str,
    gcp_json_rel: str,
    gcp_chip_rel: str,
) -> str:
    """Return a ready-to-paste SceneConfig(...) block."""
    meta_line = f'\n        metadata_json="{metadata_json}",' if metadata_json else ""
    gcp_json_line = f'\n        gcp_json="{gcp_json_rel}",' if gcp_json_rel else ""
    return f"""
    "{scene_name}": SceneConfig(
        name="{scene_name}",
        phisat_dir="{phisat_rel}",
        phisat_image="{phisat_image_rel}",{meta_line}
        sentinel_dir="{sentinel_rel}",
        dem_file="{dem_rel}",{gcp_json_line}
        gcp_chip_dir="{gcp_chip_rel}",
        tie_points_csv="outputs/{scene_name}/tie_points.csv",
        calib_json="outputs/{scene_name}/calibration.json",
        ortho_tif="outputs/{scene_name}/ortho.tif",
        initial_f=105790.0,
    ),
"""


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_fetch(config: SceneConfig, *, no_gcp_chips: bool = False) -> None:
    """
    Download Sentinel-2, DEM, and GCP data for a scene.

    Uses GL_scene_0.json from the PhiSat folder to derive the footprint.
    All downloads are skipped if the files already exist.
    """
    print("\n" + "=" * 60)
    print(f"FETCH — automatic data download — scene '{config.name}'")
    print("=" * 60)

    # ── 1. Derive footprint ────────────────────────────────────────
    phisat_dir = config.phisat_dir_path
    print(f"\n  PhiSat dir : {phisat_dir}")

    lon_min, lat_min, lon_max, lat_max = _load_footprint(phisat_dir)
    print(f"  Raw footprint : lon [{lon_min:.4f}, {lon_max:.4f}]  "
          f"lat [{lat_min:.4f}, {lat_max:.4f}]")

    # Add margin for Sentinel-2 and DEM (need coverage beyond image edges)
    s_lon_min, s_lat_min, s_lon_max, s_lat_max = _expand_bbox(
        lon_min, lat_min, lon_max, lat_max, margin_deg=0.15)

    acq_date = _load_acquisition_time(phisat_dir)
    print(f"  Acquisition date: {acq_date or 'unknown'}")

    # ── 2. Sentinel-2 ──────────────────────────────────────────────
    sentinel_dir = PROJECT_ROOT / "sentinel" / f"sentinel_{config.name}"
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    try:
        s2_path = download_sentinel(
            s_lon_min, s_lat_min, s_lon_max, s_lat_max,
            acq_date=acq_date,
            output_dir=sentinel_dir,
        )
        # Update config so later stages can find it
        config.sentinel_dir = str(
            s2_path.parent.relative_to(PROJECT_ROOT)
        )
        print(f"  Sentinel-2 dir : {config.sentinel_dir}")
    except Exception as e:
        print(f"\n  ✗ Sentinel-2 download failed: {e}")
        print("    You can download manually from https://dataspace.copernicus.eu/")

    # ── 3. DEM ─────────────────────────────────────────────────────
    dem_dir = PROJECT_ROOT / "DEM"
    dem_dir.mkdir(parents=True, exist_ok=True)
    dem_path = dem_dir / f"{config.name}.tif"

    try:
        download_dem(
            s_lon_min, s_lat_min, s_lon_max, s_lat_max,
            output_path=dem_path,
        )
        config.dem_file = str(dem_path.relative_to(PROJECT_ROOT))
        print(f"  DEM file : {config.dem_file}")
    except Exception as e:
        print(f"\n  ✗ DEM download failed: {e}")
        print("    You can download manually from https://opentopography.org/")

    # ── 4. GCPs ────────────────────────────────────────────────────
    gcp_dir = PROJECT_ROOT / "gcp" / config.name
    gcp_dir.mkdir(parents=True, exist_ok=True)

    try:
        json_paths = download_gcps(
            lon_min, lat_min, lon_max, lat_max,
            gcp_dir=gcp_dir,
            include_chips=not no_gcp_chips,
        )
        if json_paths:
            config.gcp_json = str(json_paths[0].relative_to(PROJECT_ROOT))
            config.gcp_chip_dir = str(
                (gcp_dir / "L1C_chips").relative_to(PROJECT_ROOT)
            )
            print(f"  GCP JSON  : {config.gcp_json}")
            print(f"  GCP chips : {config.gcp_chip_dir}")
    except Exception as e:
        print(f"\n  ✗ GCP download failed: {e}")
        print("    You can download manually from https://s2gri.copernicus.eu/")

    # ── 5. Print config block ───────────────────────────────────────
    _find_phisat_image(config)

    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(
        "\nAdd the following block to pipeline/config.py → SCENES dict:\n"
    )
    meta_rel = (
        config.metadata_json
        if config.metadata_json
        else None
    )
    phisat_image_rel = config.phisat_image
    print(_build_scene_config_block(
        scene_name=config.name,
        phisat_rel=config.phisat_dir,
        phisat_image_rel=phisat_image_rel,
        metadata_json=meta_rel,
        sentinel_rel=config.sentinel_dir or f"sentinel/sentinel_{config.name}",
        dem_rel=config.dem_file or f"DEM/{config.name}.tif",
        gcp_json_rel=config.gcp_json,  # None if no GCP tile was downloaded
        gcp_chip_rel=config.gcp_chip_dir or f"gcp/{config.name}/L1C_chips",
    ))


def _find_phisat_image(config: SceneConfig) -> None:
    """
    Auto-detect the PhiSat-2 RGB band file and session metadata JSON
    inside the PhiSat folder, updating config in place.

    Priority order:
      1. Any file whose name contains 'RGB'
      2. The file whose name ends in '_12_<n>.tiff' with the highest n
         (multi-band stacked image is typically numbered last)
      3. First .tiff/.tif found
    """
    bands_dir = config.phisat_dir_path / "bands"
    if bands_dir.exists():
        candidates = sorted(bands_dir.glob("*.tiff")) + sorted(bands_dir.glob("*.tif"))

        # Priority 1: file containing 'RGB' in its name
        rgb_matches = [c for c in candidates if "RGB" in c.name]
        if rgb_matches:
            chosen = rgb_matches[0]
        elif candidates:
            chosen = candidates[0]
        else:
            chosen = None

        if chosen is not None:
            config.phisat_image = str(chosen.relative_to(config.phisat_dir_path))

    # Auto-detect session metadata JSON
    if config.metadata_json is None:
        for p in config.phisat_dir_path.glob("session_*.json"):
            config.metadata_json = p.name
            break
