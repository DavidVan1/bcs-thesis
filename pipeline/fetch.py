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
    run_fetch(scene_dir)

Google Earth Engine authentication:
    Run once before first use:
        earthengine authenticate
    or inside the environment:
        python -c "import ee; ee.Authenticate()"
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Couldn't find STAC entry", category=RuntimeWarning)


import json
import logging
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import math

logger = logging.getLogger(__name__)
import geedim

# ── Constants ────────────────────────────────────────────────────────────
ESA_TCI_DIVISOR: float = 3558.0  # Hardcoded by the ESA L1C processor
GEE_ENDPOINT: str = "https://earthengine.googleapis.com"


# ═══════════════════════════════════════════════════════════════════════════
# Google Earth Engine helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_gee_initialized() -> None:
    try:
        import ee
    except ImportError:
        raise ImportError(
            "Google Earth Engine packages not found.\n"
            "Install with:  pip install earthengine-api geedim\n"
            "Then authenticate once:  earthengine authenticate --auth_mode=notebook"
        )
    try:
        ee.Initialize(opt_url=GEE_ENDPOINT)
    except ee.EEException:
        raise RuntimeError(
            "Google Earth Engine not authenticated.\n"
            "Run:  earthengine authenticate --auth_mode=notebook"
        )


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
    ratio: float = 0.06,
    min_deg: float = 0.01,
    max_deg: float = 0.50,
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
    ESA TCI formula: clamp(DN / ESA_TCI_DIVISOR * 255, 0, 255).toByte()

    Note: copyProperties() returns ee.Element (EE API quirk), so we cast
    back to ee.Image via ee.Image() to preserve the .clip() method.
    """
    import ee
    tci = (
        image.select(["B4", "B3", "B2"])
             .divide(ESA_TCI_DIVISOR).multiply(255)
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
    output_path: Path,
    *,
    date_window_days: int = 90,
    max_cloud_pct: float = 5.0,
    min_coverage: float = 0.68,
    scale_m: float = 10.0,
    region_buffer_m: float = 5000.0,
    clip_to_region: bool = False,
) -> Path:
    """
    Download the least-cloudy Sentinel-2 TCI (True Colour Image) covering
    the footprint from Google Earth Engine.

    Output: 3-band uint8 GeoTIFF (R=B4, G=B3, B=B2) equivalent to the
    official ESA TCI product — directly usable by the feature matcher.

        Selection strategy is intentionally simple and robust for reference use:
            1) search by scene center point (like manual EE inspection)
            2) rank by low cloud and AOI valid coverage
            3) export selected image clipped to AOI
    """
    out_tif = Path(output_path)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists():
        logger.info("  Sentinel-2 already exists — skipping search/download: %s", out_tif.name)
        return out_tif

    _ensure_gee_initialized()
    import ee
    from geedim.mask import BaseImage

    logger.info(
        "  Sentinel-2 TCI — bounding box: "
        "[%.4f,%.4f] x [%.4f,%.4f]",
        lon_min, lon_max, lat_min, lat_max)

    aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    region = aoi.buffer(region_buffer_m).bounds() if region_buffer_m > 0 else aoi

    logger.info(
        "  Sentinel AOI + Buffer : [%.4f,%.4f] x [%.4f,%.4f]",
        lon_min, lon_max, lat_min, lat_max,
    )
    logger.info("  Sentinel buffer      : %.0f m", region_buffer_m)

    def _find_best_image(start: str, end: str, cloud_limit: float):
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_limit))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        if collection.size().getInfo() == 0:
            return None

        candidates = collection.limit(120)
        times = candidates.aggregate_array("system:time_start").getInfo() or []

        for ts in times:
            image = ee.Image(collection.filter(ee.Filter.eq("system:time_start", ts)).first())
            date_str = ee.Date(ts).format("YYYY-MM-dd").getInfo()

            # AOI data coverage from valid-mask fraction over B4.
            valid_mask = image.select("B4").mask()
            cov_stat = valid_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=60,
                maxPixels=1e7,
                bestEffort=True,
            )
            coverage = cov_stat.get("B4").getInfo()
            coverage = float(coverage) if coverage is not None else 0.0

            cloud = float(image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo() or 0.0)
            logger.info(
                "    %s cloud %5.2f%% coverage %5.1f%%",
                date_str,
                cloud,
                coverage * 100,
            )

            if coverage >= min_coverage:
                return image

        return None

    import datetime
    best_image = None

    # ── pass 1: around acquisition date ───────────────────────────
    if acq_date:
        dt = datetime.datetime.strptime(acq_date, "%Y-%m-%d")
        td = datetime.timedelta(days=date_window_days)
        start = (dt - td).strftime("%Y-%m-%d")
        end   = (dt + td).strftime("%Y-%m-%d")
        logger.info("  Pass 1 — %s -> %s (cloud <= %.0f%%)", start, end, max_cloud_pct)
        best_image = _find_best_image(start, end, max_cloud_pct)

    # ── pass 2: full archive ───────────────────────────────────────
    if best_image is None:
        logger.info("  Pass 2 — full archive (cloud <= %.0f%%)", max_cloud_pct)
        best_image = _find_best_image("2015-01-01", "2099-01-01", max_cloud_pct)

    # ── pass 3: relaxed cloud ──────────────────────────────────────
    if best_image is None:
        logger.info("  Pass 3 — full archive, cloud <= 30%%")
        best_image = _find_best_image("2015-01-01", "2099-01-01", 30.0)

    # ── pass 4: no cloud filter ────────────────────────────────────
    if best_image is None:
        logger.info("  Pass 4 — full archive, no cloud filter (coverage >= %.0f%%)", min_coverage * 100)
        best_image = _find_best_image("2015-01-01", "2099-01-01", 100.0)
    # Last resort: allow lower coverage rather than failing hard.
    if best_image is None and min_coverage > 0.50:
        relaxed = 0.50
        logger.warning(
            "  No candidate met %.0f%% coverage; retrying with %.0f%%",
            min_coverage * 100,
            relaxed * 100,
        )
        min_coverage = relaxed
        best_image = _find_best_image("2015-01-01", "2099-01-01", 100.0)

    if best_image is None:
        raise RuntimeError("No Sentinel-2 images found for the given footprint.")

    best_date = ee.Date(best_image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    cloud_frac = float(best_image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo() or 0.0)

    logger.info("  Selected date       : %s", best_date)
    logger.info("  Cloud (scene-level) : %.2f%%", cloud_frac)
    logger.info("  Selected date used for export: %s", best_date)

    native_crs = best_image.select("B4").projection().crs().getInfo()
    export_crs = "EPSG:4326"

    logger.info("  Using AOI-selected image for %s", best_date)
    logger.info("  Sentinel source CRS  : %s", native_crs)
    logger.info("  Sentinel export CRS  : %s", export_crs)
    tci_image = _make_tci(best_image)
    gd_img = BaseImage(tci_image)

    logger.info("  Downloading TCI to %s ...", out_tif)
    # if clip_to_region:
    #     logger.info("  Export mode         : clipped to AOI+buffer")
    #     prep_img = tci_image.gd.prepareForExport(
    #         region=region,
    #         scale=scale_m,
    #         crs=native_crs,
    #         dtype="uint8"
    #     )
    #     prep_img.gd.toGeoTIFF(str(out_tif), overwrite=True)
    # else:
    #     logger.info("  Export mode         : full Sentinel scene (no clipping)")
    #     prep_img = tci_image.gd.prepareForExport(
    #         scale=scale_m,
    #         crs=native_crs,
    #         dtype="uint8"
    #     )
    #     prep_img.gd.toGeoTIFF(str(out_tif), overwrite=True)
    if clip_to_region:
        logger.info("  Export mode         : clipped to AOI+buffer")
        gd_img.download(
            str(out_tif),
            region=region,
            scale=scale_m,
            crs=native_crs,
            dtype="uint8",
            overwrite=True,
        )
    else:
        logger.info("  Export mode         : full Sentinel scene (no clipping)")
        gd_img.download(
            str(out_tif),
            scale=scale_m,
            crs=native_crs,
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
    region_buffer_m: float = 60000.0,
) -> Path:
    """
    Download Copernicus GLO-30 DEM for the footprint from Google Earth Engine.

    The result is a single GeoTIFF in EPSG:4326 at ~30 m resolution.
    """
    _ensure_gee_initialized()
    import ee
    from geedim.mask import BaseImage

    logger.info(
        "  Copernicus DEM GLO-30 — bbox: "
        "[%.4f,%.4f] × [%.4f,%.4f]",
        lon_min, lon_max, lat_min, lat_max)

    output_path = Path(output_path)
    if output_path.exists():
        logger.info("  Already exists — skipping download: %s", output_path.name)
        return output_path

    aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    region = aoi.buffer(region_buffer_m).bounds() if region_buffer_m > 0 else aoi
    logger.info("  DEM buffer          : %.0f m", region_buffer_m)

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
    
    
    # prep_dem = dem.gd.prepareForExport(
    #     region=region,
    #     scale=scale_m,
    #     crs="EPSG:4326",
    #     dtype="float32"
    # )
    # prep_dem.gd.toGeoTIFF(str(output_path), overwrite=True)

    logger.info("  Saved: %s", output_path)
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
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_fetch(scene_dir: Path, *,
              no_gcp_chips: bool = False) -> None:
    """
    Download Sentinel-2, DEM, and GCP data for a scene.

    Uses GL_scene_0.json from the PhiSat folder to derive the footprint.
    All downloads are skipped if the files already exist.
    """
    scene_dir = Path(scene_dir)
    logger.info("=" * 60)
    logger.info("FETCH — automatic data download — scene '%s'", scene_dir.name)
    logger.info("=" * 60)

    # ── 1. Derive footprint ────────────────────────────────────────
    phisat_dir = scene_dir
    logger.info("  PhiSat dir : %s", phisat_dir)

    lon_min, lat_min, lon_max, lat_max = _load_footprint(phisat_dir)
    logger.info(
        "  Raw footprint : lon [%.4f, %.4f]  lat [%.4f, %.4f]",
        lon_min, lon_max, lat_min, lat_max)

    # Keep fetch AOI modest for stable reference-image selection.
    # Matching margin is handled later during reprojection, not during fetch.
    fetch_margin = _adaptive_fetch_margin(lon_min, lat_min, lon_max, lat_max)
    logger.info("  Fetch margin   : %.4f°", fetch_margin)
    s_lon_min, s_lat_min, s_lon_max, s_lat_max = _expand_bbox(
        lon_min, lat_min, lon_max, lat_max, margin_deg=fetch_margin)

    acq_date = _load_acquisition_time(phisat_dir)
    logger.info("  Acquisition date: %s", acq_date or 'unknown')

    # ── 2. Sentinel-2 ──────────────────────────────────────────────
    try:
        sentinel_path = scene_dir / "sentinel.tif"
        s2_path = download_sentinel(
            s_lon_min, s_lat_min, s_lon_max, s_lat_max,
            acq_date=acq_date,
            output_path=sentinel_path,
        )
        logger.info("  Sentinel-2 file: %s", s2_path)
    except Exception as e:
        logger.error("  Sentinel-2 download failed: %s", e)
        logger.error("    You can download manually from https://dataspace.copernicus.eu/")

    # ── 3. DEM ─────────────────────────────────────────────────────
    dem_path = scene_dir / "dem.tif"

    try:
        download_dem(
            s_lon_min, s_lat_min, s_lon_max, s_lat_max,
            output_path=dem_path,
        )
        logger.info("  DEM file : %s", dem_path)
    except Exception as e:
        logger.error("  DEM download failed: %s", e)
        logger.error("    You can download manually from https://opentopography.org/")

    # ── 4. GCPs ────────────────────────────────────────────────────
    gcp_dir = scene_dir / "sentinel_gri"
    gcp_dir.mkdir(parents=True, exist_ok=True)

    try:
        json_paths = download_gcps(
            lon_min, lat_min, lon_max, lat_max,
            gcp_dir=gcp_dir,
            include_chips=not no_gcp_chips,
        )
        if json_paths:
            logger.info("  GCP JSON  : %s", json_paths[0])
            logger.info("  GCP chips : %s", gcp_dir / "L1C_chips")
    except Exception as e:
        logger.error("  GCP download failed: %s", e)
        logger.error("    You can download manually from https://s2gri.copernicus.eu/")

    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)


def _find_phisat_image(scene_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Auto-detect the PhiSat-2 RGB band file and session metadata JSON
    inside the PhiSat folder.

    Priority order:
      1. Any file whose name contains 'RGB'
      2. The file whose name ends in '_12_<n>.tiff' with the highest n
         (multi-band stacked image is typically numbered last)
      3. First .tiff/.tif found
    """
    bands_dir = scene_dir / "bands"
    chosen: Optional[Path] = None
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

    metadata_path = None
    for p in scene_dir.glob("session_*.json"):
        metadata_path = p
        break

    return chosen, metadata_path
