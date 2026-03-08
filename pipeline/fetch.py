"""
Automatic data fetching for a PhiSat-2 scene.

Given only the phisat folder (AOCS.json + GL_scene_0.json), downloads:

  1. Sentinel-2 L1C imagery  – Google Earth Engine (COPERNICUS/S2_HARMONIZED)
  2. Copernicus DEM GLO-30    – Google Earth Engine (COPERNICUS/DEM/GLO30)
  3. Copernicus GCP database  – ESA S2 GRI HTTPS tiles
  4. US national ortho (optional, US only) – USDA NAIP via Google Earth Engine

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
import logging
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

logger = logging.getLogger(__name__)


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


def _adaptive_fetch_margin(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    *,
    ratio: float = 0.10,
    min_deg: float = 0.01,
    max_deg: float = 0.05,
) -> float:
    """
    Compute a scene-size-aware margin (degrees) for fetch-time AOIs.

    A fixed large margin can excessively enlarge the Sentinel search region,
    causing slow date scoring and very low coverage percentages. This helper
    keeps a small proportional pad while clamping to sensible bounds.
    """
    span_lon = max(0.0, lon_max - lon_min)
    span_lat = max(0.0, lat_max - lat_min)
    span = max(span_lon, span_lat)
    margin = span * ratio
    return max(min_deg, min(max_deg, margin))


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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_tci = sorted(out_dir.glob("sentinel_TCI_*.tif"))
    if existing_tci:
        chosen = existing_tci[-1]
        logger.info("  Sentinel-2 already exists — skipping search/download: %s", chosen.name)
        return chosen

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

    logger.info(
        "  Sentinel-2 TCI — bounding box: "
        "[%.4f,%.4f] x [%.4f,%.4f]",
        lon_min, lon_max, lat_min, lat_max)

    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    def _best_date(start: str, end: str, cloud_limit: float,
                   min_coverage: float = 0.80):
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

        logger.info("    %d granule(s) in window — scoring ROI cloud + coverage ...", size)

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
            logger.info("    %s  cloud %5.2f%%  coverage %5.1f%%  (%d granule(s))",
                        date_str, frac, coverage * 100, n_gran)

            if coverage >= min_coverage:
                return date_str, frac

        logger.info(
            "    No date with cloud < %.0f%% and coverage >= %.0f%%",
            cloud_limit, min_coverage * 100)
        return None, None

    # ── pass 1: around acquisition date ───────────────────────────
    import datetime
    best_date = cloud_frac = None
    if acq_date:
        dt = datetime.datetime.strptime(acq_date, "%Y-%m-%d")
        td = datetime.timedelta(days=date_window_days)
        start = (dt - td).strftime("%Y-%m-%d")
        end   = (dt + td).strftime("%Y-%m-%d")
        logger.info("  Pass 1 — %s -> %s  (ROI cloud <= %.0f%%)", start, end, max_cloud_pct)
        best_date, cloud_frac = _best_date(start, end, max_cloud_pct)

    # ── pass 2: full archive ───────────────────────────────────────
    if best_date is None:
        logger.info("  Pass 2 — full archive  (ROI cloud <= %.0f%%)", max_cloud_pct)
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", max_cloud_pct)

    # ── pass 3: relaxed cloud ──────────────────────────────────────
    if best_date is None:
        logger.info("  Pass 3 — full archive, cloud <= 30%%")
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", 30.0)

    # ── pass 4: no cloud filter ────────────────────────────────────
    if best_date is None:
        logger.info("  Pass 4 — full archive, no cloud filter")
        best_date, cloud_frac = _best_date("2015-01-01", "2099-01-01", 100.0)
        if best_date is None:
            raise RuntimeError("No Sentinel-2 images found for the given footprint.")

    logger.info("  Selected date       : %s", best_date)
    logger.info("  Cloud (footprint)   : %.2f%%", cloud_frac)

    out_tif = out_dir / f"sentinel_TCI_{best_date}.tif"

    if out_tif.exists():
        logger.info("  Already exists — skipping download: %s", out_tif.name)
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
        logger.warning("  L1C collection empty for %s — using SR collection for TCI", best_date)
        l1c_day = _tci_col("COPERNICUS/S2_SR_HARMONIZED")

    n_granules = l1c_day.size().getInfo()
    logger.info("  Mosaicking %d granule(s) for %s", n_granules, best_date)
    tci_mosaic = _make_tci(l1c_day.mosaic()).clip(region)
    gd_img = BaseImage(tci_mosaic)

    logger.info("  Downloading TCI to %s ...", out_tif)
    gd_img.download(
        str(out_tif),
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        dtype="uint8",
        overwrite=True,
    )
    logger.info("  Saved: %s", out_tif)
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

    logger.info(
        "  Copernicus DEM GLO-30 — bbox: "
        "[%.4f,%.4f] × [%.4f,%.4f]",
        lon_min, lon_max, lat_min, lat_max)

    output_path = Path(output_path)
    if output_path.exists():
        logger.info("  Already exists — skipping download: %s", output_path.name)
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

    logger.info("  Downloading to %s ...", output_path)
    gd_img.download(
        str(output_path),
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        dtype="float32",
        overwrite=True,
    )
    logger.info("  Saved: %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# US national mapping (NAIP) via Google Earth Engine
# ═══════════════════════════════════════════════════════════════════════════

def _bbox_is_us(lon_min: float, lat_min: float,
                lon_max: float, lat_max: float) -> bool:
    """Heuristic test for U.S. coverage (CONUS + Alaska + Hawaii + PR)."""
    c_lon = 0.5 * (lon_min + lon_max)
    c_lat = 0.5 * (lat_min + lat_max)
    return (-170.0 <= c_lon <= -60.0) and (18.0 <= c_lat <= 72.0)


def download_us_national_ortho(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    acq_date: Optional[str],
    output_dir: Path,
    *,
    scale_m: float = 2.0,
    max_raw_gb: float = 2.5,
) -> Optional[Path]:
    """
    Download a USDA NAIP ortho mosaic (RGB) for the AOI.

    NAIP is a free U.S. national aerial ortho source, suitable as an
    independent high-resolution reference layer for validation.
    """
    if not _bbox_is_us(lon_min, lat_min, lon_max, lat_max):
        logger.info("  US national mapping skipped (footprint outside U.S.).")
        return None

    try:
        import ee
        from geedim.mask import BaseImage
    except ImportError:
        logger.info("  US national mapping skipped: missing earthengine-api/geedim")
        return None

    try:
        ee.Initialize(opt_url="https://earthengine.googleapis.com")
    except ee.EEException:
        logger.info("  US national mapping skipped: Earth Engine not authenticated")
        return None

    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    # Estimate raw RGB uint8 size and auto-coarsen scale when too large.
    # This avoids accidental 10+ GB exports for big AOIs at 1 m.
    lat_c = 0.5 * (lat_min + lat_max)
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * max(math.cos(math.radians(lat_c)), 1e-6)
    width_m = max((lon_max - lon_min) * m_per_deg_lon, 1.0)
    height_m = max((lat_max - lat_min) * m_per_deg_lat, 1.0)

    eff_scale = float(scale_m)
    raw_bytes = (width_m / eff_scale) * (height_m / eff_scale) * 3.0
    raw_gb = raw_bytes / 1e9
    if raw_gb > max_raw_gb:
        factor = math.sqrt(raw_gb / max_raw_gb)
        eff_scale *= factor
        raw_bytes = (width_m / eff_scale) * (height_m / eff_scale) * 3.0
        raw_gb = raw_bytes / 1e9
        logger.warning(
            "  NAIP AOI is large; auto-adjusting scale to %.2f m "
            "(estimated raw %.2f GB)", eff_scale, raw_gb)

    # Prefer acquisition-year ±2y; fallback to full archive latest mosaic.
    naip = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(region)

    import datetime
    year_tag = "latest"
    if acq_date:
        try:
            y = datetime.datetime.strptime(acq_date, "%Y-%m-%d").year
            start = f"{max(2003, y-2)}-01-01"
            end = f"{y+2}-12-31"
            cand = naip.filterDate(start, end)
            if cand.size().getInfo() > 0:
                naip = cand
                year_tag = f"{max(2003, y-2)}_{y+2}"
        except Exception:
            pass

    count = naip.size().getInfo()
    if count == 0:
        logger.info("  US national mapping skipped: no NAIP imagery found for AOI")
        return None

    # Cloud-free aerial ortho; use median to smooth seam differences.
    img = naip.select(["R", "G", "B"]).median().clip(region).toUint8()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_tif = output_dir / f"us_naip_{year_tag}.tif"
    if out_tif.exists():
        logger.info("  US national ortho already exists — skipping: %s", out_tif.name)
        return out_tif

    logger.info("  US national ortho (NAIP): %d tile(s) -> %s", count, out_tif.name)
    gd_img = BaseImage(img)
    gd_img.download(
        str(out_tif),
        region=region,
        scale=eff_scale,
        crs="EPSG:4326",
        dtype="uint8",
        overwrite=True,
    )
    logger.info("  Saved US national ortho: %s", out_tif)
    return out_tif


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
    logger.info("  GCP tiles needed: %s", tiles)

    gcp_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = gcp_dir / "L1C_chips"
    if include_chips:
        chip_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []

    for tile in tiles:
        json_dest = gcp_dir / f"{tile}.json"

        # Check if already fully downloaded
        if json_dest.exists():
            logger.info("  %s — already exists, skipping", tile)
            json_paths.append(json_dest)
            continue

        url = _wasabi_gri_url(tile)
        logger.info("  Downloading %s from Wasabi S3 ...", tile)
        logger.info("    %s", url)

        # Stream tar.gz into a temp file, then extract
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if not _try_download_url(url, tmp_path):
                logger.warning("    Could not download %s_L1C.tar.gz", tile)
                logger.warning("    Manual download: %s", url)
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
                logger.info(
                    "  %s.json extracted%s", tile,
                    f"  ({n_chips} chips)" if include_chips else "")
                json_paths.append(json_dest)
            else:
                logger.warning("  %s.json not found inside archive", tile)

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
    us_national_ortho_rel: Optional[str],
) -> str:
    """Return a ready-to-paste SceneConfig(...) block."""
    meta_line = f'\n        metadata_json="{metadata_json}",' if metadata_json else ""
    gcp_json_line = f'\n        gcp_json="{gcp_json_rel}",' if gcp_json_rel else ""
    us_mapping_line = (f'\n        us_national_ortho="{us_national_ortho_rel}",'
                       if us_national_ortho_rel else "")
    return f"""
    "{scene_name}": SceneConfig(
        name="{scene_name}",
        phisat_dir="{phisat_rel}",
        phisat_image="{phisat_image_rel}",{meta_line}
        sentinel_dir="{sentinel_rel}",
        dem_file="{dem_rel}",{gcp_json_line}{us_mapping_line}
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

def run_fetch(config: SceneConfig, *,
              no_gcp_chips: bool = False,
              fetch_us_mapping: bool = True) -> None:
    """
    Download Sentinel-2, DEM, and GCP data for a scene.

    Uses GL_scene_0.json from the PhiSat folder to derive the footprint.
    All downloads are skipped if the files already exist.
    """
    logger.info("=" * 60)
    logger.info("FETCH — automatic data download — scene '%s'", config.name)
    logger.info("=" * 60)

    # ── 1. Derive footprint ────────────────────────────────────────
    phisat_dir = config.phisat_dir_path
    logger.info("  PhiSat dir : %s", phisat_dir)

    lon_min, lat_min, lon_max, lat_max = _load_footprint(phisat_dir)
    logger.info(
        "  Raw footprint : lon [%.4f, %.4f]  lat [%.4f, %.4f]",
        lon_min, lon_max, lat_min, lat_max)

    # Add a small adaptive margin for Sentinel-2 and DEM.
    # A fixed 0.15° margin can over-expand compact scenes and slow S2 scoring.
    fetch_margin = _adaptive_fetch_margin(lon_min, lat_min, lon_max, lat_max)
    logger.info("  Fetch margin   : %.4f°", fetch_margin)
    s_lon_min, s_lat_min, s_lon_max, s_lat_max = _expand_bbox(
        lon_min, lat_min, lon_max, lat_max, margin_deg=fetch_margin)

    acq_date = _load_acquisition_time(phisat_dir)
    logger.info("  Acquisition date: %s", acq_date or 'unknown')

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
        logger.info("  Sentinel-2 dir : %s", config.sentinel_dir)
    except Exception as e:
        logger.error("  Sentinel-2 download failed: %s", e)
        logger.error("    You can download manually from https://dataspace.copernicus.eu/")

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
        logger.info("  DEM file : %s", config.dem_file)
    except Exception as e:
        logger.error("  DEM download failed: %s", e)
        logger.error("    You can download manually from https://opentopography.org/")

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
            logger.info("  GCP JSON  : %s", config.gcp_json)
            logger.info("  GCP chips : %s", config.gcp_chip_dir)
    except Exception as e:
        logger.error("  GCP download failed: %s", e)
        logger.error("    You can download manually from https://s2gri.copernicus.eu/")

    # ── 5. US national mapping (optional, US-only) ─────────────────
    if fetch_us_mapping:
        us_out_dir = PROJECT_ROOT / "national" / f"us_{config.name}"
        # NAIP fetch does not need the large Sentinel/DEM margin.
        n_lon_min, n_lat_min, n_lon_max, n_lat_max = _expand_bbox(
            lon_min, lat_min, lon_max, lat_max, margin_deg=0.02)
        try:
            us_path = download_us_national_ortho(
                n_lon_min, n_lat_min, n_lon_max, n_lat_max,
                acq_date=acq_date,
                output_dir=us_out_dir,
            )
            if us_path is not None:
                config.us_national_ortho = str(us_path.relative_to(PROJECT_ROOT))
                logger.info("  US national ortho : %s", config.us_national_ortho)
        except Exception as e:
            logger.error("  US national mapping fetch failed: %s", e)
            logger.info("    Continuing without US national ortho.")

    # ── 6. Print config block ───────────────────────────────────────
    _find_phisat_image(config)

    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "\nAdd the following block to pipeline/config.py \u2192 SCENES dict:\n"
    )
    meta_rel = (
        config.metadata_json
        if config.metadata_json
        else None
    )
    phisat_image_rel = config.phisat_image
    logger.info(_build_scene_config_block(
        scene_name=config.name,
        phisat_rel=config.phisat_dir,
        phisat_image_rel=phisat_image_rel,
        metadata_json=meta_rel,
        sentinel_rel=config.sentinel_dir or f"sentinel/sentinel_{config.name}",
        dem_rel=config.dem_file or f"DEM/{config.name}.tif",
        gcp_json_rel=config.gcp_json,  # None if no GCP tile was downloaded
        gcp_chip_rel=config.gcp_chip_dir or f"gcp/{config.name}/L1C_chips",
        us_national_ortho_rel=config.us_national_ortho,
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
