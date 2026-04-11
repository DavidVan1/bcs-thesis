"""
DEM-aware orthorectification of PhiSat-2 pushbroom imagery.

Pipeline:
  1. Estimate image footprint from DEM-grid RPC validity.
  2. Build a sparse lookup table (LUT) via RPC ground-to-image projection.
  3. Interpolate LUT to full resolution.
  4. Resample raw image with cv2.remap.
  5. Write orthorectified GeoTIFF.
"""

import logging
import os
import numpy as np
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from pyproj import Transformer

import rasterio

from .config import SceneConfig, PHISAT_GSD_M

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────
TARGET_GSD_M: float = PHISAT_GSD_M     # output ground sample distance
FOOTPRINT_MARGIN_DEG: float = 0.02     # degrees padding around footprint
LUT_MIN_STEP: int = 10
LUT_GRID_DIVISOR: int = 100             # max(dim) // this = step


def _gdal_subprocess_env() -> dict:
    """Return environment for GDAL subprocesses with explicit PROJ/GDAL data paths."""
    env = os.environ.copy()

    candidates = []
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix))
    candidates.append(Path("/home/van6/miniconda3/envs/phisat"))

    for prefix in candidates:
        proj_dir = prefix / "share" / "proj"
        if proj_dir.exists():
            env.setdefault("PROJ_LIB", str(proj_dir))
            env.setdefault("PROJ_DATA", str(proj_dir))
            break

    for prefix in candidates:
        gdal_dir = prefix / "share" / "gdal"
        if gdal_dir.exists():
            env.setdefault("GDAL_DATA", str(gdal_dir))
            break

    return env


# ═══════════════════════════════════════════════════════════════════════════

def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Return EPSG code for the UTM zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _rpc_to_gdal_metadata(rpc_payload: dict) -> dict:
    """Convert generic RPC dict into GDAL RPC metadata-domain key/value pairs."""

    def _pick(*keys):
        for key in keys:
            if key in rpc_payload:
                return rpc_payload[key]
        raise KeyError(f"Missing RPC key; expected one of: {keys}")

    def _scalar(*keys) -> str:
        value = _pick(*keys)
        return str(float(value))

    def _coeff(*keys) -> str:
        value = _pick(*keys)
        if not isinstance(value, (list, tuple)) or len(value) != 20:
            raise ValueError(f"RPC coefficient vector must have length 20 for keys: {keys}")
        return " ".join(str(float(v)) for v in value)

    return {
        "LINE_OFF": _scalar("line_off", "LINE_OFF"),
        "SAMP_OFF": _scalar("samp_off", "SAMP_OFF", "sample_off", "SAMPLE_OFF"),
        "LAT_OFF": _scalar("lat_off", "LAT_OFF"),
        "LONG_OFF": _scalar("lon_off", "LONG_OFF", "long_off"),
        "HEIGHT_OFF": _scalar("height_off", "HEIGHT_OFF", "alt_off", "ALT_OFF"),
        "LINE_SCALE": _scalar("line_scale", "LINE_SCALE"),
        "SAMP_SCALE": _scalar("samp_scale", "SAMP_SCALE", "sample_scale", "SAMPLE_SCALE"),
        "LAT_SCALE": _scalar("lat_scale", "LAT_SCALE"),
        "LONG_SCALE": _scalar("lon_scale", "LONG_SCALE", "long_scale"),
        "HEIGHT_SCALE": _scalar("height_scale", "HEIGHT_SCALE", "alt_scale", "ALT_SCALE"),
        "LINE_NUM_COEFF": _coeff("line_num", "LINE_NUM_COEFF"),
        "LINE_DEN_COEFF": _coeff("line_den", "LINE_DEN_COEFF"),
        "SAMP_NUM_COEFF": _coeff("samp_num", "SAMP_NUM_COEFF", "sample_num", "SAMPLE_NUM_COEFF"),
        "SAMP_DEN_COEFF": _coeff("samp_den", "SAMP_DEN_COEFF", "sample_den", "SAMPLE_DEN_COEFF"),
    }


def _build_rpc_vrt(src_path: Path, vrt_path: Path, rpc_payload: dict) -> None:
    """Create a temporary VRT from source and inject RPC metadata domain."""
    gdal_env = _gdal_subprocess_env()
    subprocess.run(
        ["gdal_translate", "-of", "VRT", str(src_path), str(vrt_path)],
        check=True,
        env=gdal_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    tree = ET.parse(vrt_path)
    root = tree.getroot()

    for node in list(root.findall("Metadata")):
        if node.attrib.get("domain") == "RPC":
            root.remove(node)

    rpc_md = ET.Element("Metadata", {"domain": "RPC"})
    for key, value in _rpc_to_gdal_metadata(rpc_payload).items():
        mdi = ET.SubElement(rpc_md, "MDI", {"key": key})
        mdi.text = value

    root.append(rpc_md)
    tree.write(vrt_path, encoding="UTF-8", xml_declaration=True)


def _gdal_rpc_orthorectify(config: SceneConfig, rpc_payload: dict) -> Optional[str]:
    """Orthorectify with GDAL RPC warp (+ DEM). Returns output path on success."""
    if shutil.which("gdalwarp") is None or shutil.which("gdal_translate") is None:
        logger.warning("GDAL tools (gdalwarp/gdal_translate) not found; falling back to Python orthorectify")
        return None

    out_path = Path(config.ortho_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(config.dem_path)) as dem_src:
        dem_crs = dem_src.crs
        bounds = dem_src.bounds

    center_x = 0.5 * (bounds.left + bounds.right)
    center_y = 0.5 * (bounds.bottom + bounds.top)

    if dem_crs is None:
        raise ValueError("DEM CRS is missing; cannot derive target UTM CRS")

    if str(dem_crs).upper() != "EPSG:4326":
        to_wgs84 = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs84.transform(center_x, center_y)
    else:
        center_lon, center_lat = center_x, center_y

    utm_epsg = _utm_epsg_from_lonlat(center_lon, center_lat)
    logger.info("GDAL RPC warp target CRS: EPSG:%d", utm_epsg)

    gdal_env = _gdal_subprocess_env()

    with tempfile.TemporaryDirectory(prefix="rpc_ortho_") as tmp_dir:
        vrt_path = Path(tmp_dir) / "source_with_rpc.vrt"
        _build_rpc_vrt(Path(config.phisat_image_path), vrt_path, rpc_payload)

        cmd = [
            "gdalwarp",
            "-overwrite",
            "-rpc",
            "-to", f"RPC_DEM={config.dem_path}",
            "-t_srs", f"EPSG:{utm_epsg}",
            "-tr", str(TARGET_GSD_M), str(TARGET_GSD_M),
            "-tap",
            "-r", "cubic",
            "-dstnodata", "0",
            "-multi",
            "-wo", "NUM_THREADS=ALL_CPUS",
            str(vrt_path),
            str(out_path),
        ]

        logger.info("Running GDAL RPC orthorectification...")
        result = subprocess.run(
            cmd,
            env=gdal_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("GDAL RPC orthorectify failed (%d): %s", result.returncode, result.stderr.strip())
            return None

    if not out_path.exists():
        logger.warning("GDAL reported success but output missing: %s", out_path)
        return None

    logger.info("GDAL RPC orthorectification complete → %s", out_path)
    return str(out_path)


def run_orthorectify_rpc(config: SceneConfig,
                         rpc_json_path: str) -> Optional[str]:
    """Run orthorectification for a scene using direct RPC ground->image."""
    missing = []
    if config.phisat_image_path is None or not config.phisat_image_path.exists():
        missing.append(f"PhiSat image: {config.phisat_image_path}")
    # if config.aocs_path is None or not config.aocs_path.exists():
    #     missing.append(f"AOCS: {config.aocs_path}")
    if config.dem_path is None or not config.dem_path.exists():
        missing.append(f"DEM: {config.dem_path}")
    if missing:
        raise FileNotFoundError(
            "Missing files for orthorectify:\n  " + "\n  ".join(missing))

    rpc_path = Path(rpc_json_path)
    if not rpc_path.exists():
        raise FileNotFoundError(f"RPC JSON not found: {rpc_path}")

    logger.info("=" * 60)
    logger.info("ORTHORECTIFY (RPC) — scene '%s'", config.name)
    logger.info("=" * 60)

    with open(rpc_path) as f:
        rpc_payload = json.load(f)
    if isinstance(rpc_payload, dict) and "rpc" in rpc_payload and isinstance(rpc_payload["rpc"], dict):
        rpc_payload = rpc_payload["rpc"]

    out = _gdal_rpc_orthorectify(config, rpc_payload)
    if out is None:
        raise RuntimeError("GDAL RPC orthorectification failed")
    return out