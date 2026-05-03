import logging
import numpy as np
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from pyproj import Transformer
import rasterio

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
TARGET_GSD_M: float = 4.75


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Returns the EPSG code for the UTM zone containing the given (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    return (32600 + zone) if lat >= 0 else (32700 + zone)

def _parse_rpc_txt(txt_path: Path) -> dict:
    """
    Parses an RPC text file (PhiSat-2 format) into a dictionary.
    Expected format per line: KEY: VALUE(S)
    """
    rpc_data = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # If the key contains 'COEFF', it is a vector of 20 coefficients
            if 'COEFF' in key:
                rpc_data[key] = [float(x) for x in value.split()]
            else:
                # Otherwise, try to parse it as a simple scalar (offsets, scales)
                try:
                    rpc_data[key] = float(value)
                except ValueError:
                    continue
    return rpc_data

def _rpc_to_gdal_metadata(rpc_payload: dict) -> dict:
    """
    Maps the loaded dictionary keys to the standard metadata 
    domain keys expected by GDAL's RPC mechanism.
    """
    def _scalar(key) -> str:
        if key not in rpc_payload:
            raise KeyError(f"Missing required RPC key in file: {key}")
        return str(float(rpc_payload[key]))

    def _coeff(key) -> str:
        value = rpc_payload.get(key)
        if not value or len(value) != 20:
            raise ValueError(f"RPC coefficient {key} must have exactly 20 elements.")
        return " ".join(str(float(v)) for v in value)

    return {
        "LINE_OFF": _scalar("LINE_OFF"),
        "SAMP_OFF": _scalar("SAMP_OFF"),
        "LAT_OFF": _scalar("LAT_OFF"),
        "LONG_OFF": _scalar("LONG_OFF"),
        "HEIGHT_OFF": _scalar("HEIGHT_OFF"),
        "LINE_SCALE": _scalar("LINE_SCALE"),
        "SAMP_SCALE": _scalar("SAMP_SCALE"),
        "LAT_SCALE": _scalar("LAT_SCALE"),
        "LONG_SCALE": _scalar("LONG_SCALE"),
        "HEIGHT_SCALE": _scalar("HEIGHT_SCALE"),
        "LINE_NUM_COEFF": _coeff("LINE_NUM_COEFF"),
        "LINE_DEN_COEFF": _coeff("LINE_DEN_COEFF"),
        "SAMP_NUM_COEFF": _coeff("SAMP_NUM_COEFF"),
        "SAMP_DEN_COEFF": _coeff("SAMP_DEN_COEFF"),
    }

def _build_rpc_vrt(src_path: Path, vrt_path: Path, rpc_payload: dict) -> None:
    """
    Creates a temporary VRT from the source TIFF and injects 
    the RPC metadata domain into the XML.
    """
    subprocess.run(
        ["gdal_translate", "-of", "VRT", str(src_path), str(vrt_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    tree = ET.parse(vrt_path)
    root = tree.getroot()
    
    # Remove any existing RPC metadata nodes
    for node in list(root.findall("Metadata")):
        if node.attrib.get("domain") == "RPC":
            root.remove(node)

    # Append the new RPC metadata domain
    rpc_md = ET.Element("Metadata", {"domain": "RPC"})
    for key, value in _rpc_to_gdal_metadata(rpc_payload).items():
        mdi = ET.SubElement(rpc_md, "MDI", {"key": key})
        mdi.text = value

    root.append(rpc_md)
    tree.write(vrt_path, encoding="UTF-8", xml_declaration=True)

def _gdal_rpc_orthorectify(
    phisat_tiff: Path, dem_path: Path, output_path: Path, rpc_payload: dict,
) -> Optional[str]:
    """Performs the actual orthorectification using gdalwarp + RPC + DEM."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine UTM zone from the DEM center
    with rasterio.open(str(dem_path)) as dem_src:
        dem_crs = dem_src.crs
        bounds = dem_src.bounds
    
    center_x = 0.5 * (bounds.left + bounds.right)
    center_y = 0.5 * (bounds.bottom + bounds.top)

    if str(dem_crs).upper() != "EPSG:4326":
        to_wgs84 = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(center_x, center_y)
    else:
        lon, lat = center_x, center_y

    utm_epsg = _utm_epsg_from_lonlat(lon, lat)

    with tempfile.TemporaryDirectory(prefix="rpc_ortho_") as tmp_dir:
        vrt_path = Path(tmp_dir) / "source_with_rpc.vrt"
        _build_rpc_vrt(Path(phisat_tiff), vrt_path, rpc_payload)

        # Build the gdalwarp command
        cmd = [
            "gdalwarp", "-overwrite", "-rpc",
            "-to", f"RPC_DEM={dem_path}",
            "-t_srs", f"EPSG:{utm_epsg}",
            "-tr", str(TARGET_GSD_M), str(TARGET_GSD_M),
            "-tap", "-r", "cubic", "-dstnodata", "0",
            "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
            str(vrt_path), str(out_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("GDAL Error: %s", result.stderr)
            return None

    return str(out_path) if out_path.exists() else None

def run_orthorectify(
    phisat_tiff: Path,
    dem_path: Path,
    rpc_path: Path,
    output_path: Path,
) -> Path:
    """Main function – handles file validation, RPC parsing, and processing."""
    for p in [phisat_tiff, dem_path, rpc_path]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    logger.info("-" * 50)
    logger.info(f"Processing Image: {phisat_tiff.name}")
    logger.info(f"Using RPC File: {rpc_path.name}")
    logger.info("-" * 50)

    # Load RPC data from the text file
    rpc_payload = _parse_rpc_txt(rpc_path)

    # Execute orthorectification
    out = _gdal_rpc_orthorectify(phisat_tiff, dem_path, output_path, rpc_payload)
    
    if out is None:
        raise RuntimeError("Orthorectification failed (check GDAL output).")
    
    logger.info(f"Successfully created: {out}")
    return Path(out)