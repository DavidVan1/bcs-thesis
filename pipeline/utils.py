"""
Shared utilities: image I/O, enhancement, tie-point loading.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
import rasterio


# ── Tie-point I/O ───────────────────────────────────────────────────────

def load_tie_points(csv_path: str) -> List[Dict]:
    """
    Load tie points from CSV with columns: phisat_x, phisat_y, lon, lat.
    Returns list of dicts.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Tie points CSV not found: {path}")

    tie_points = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tie_points.append({
                "phisat_x": float(row["phisat_x"]),
                "phisat_y": float(row["phisat_y"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
            })
    return tie_points


def save_tie_points(tie_points: List[Dict], csv_path: str,
                    extra_fields: Optional[List[str]] = None) -> None:
    """Save tie points to CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["phisat_x", "phisat_y", "lon", "lat"]
    if extra_fields:
        fields.extend(extra_fields)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tp in tie_points:
            writer.writerow({k: tp[k] for k in fields if k in tp})


# ── Calibration JSON I/O ────────────────────────────────────────────────

def load_calibration(json_path: str) -> Dict:
    """
    Load calibration JSON and return a flat dict with keys:
        f, cx, cy, k1, k2, roll, pitch, yaw, time_shift, cx_rate
    """
    with open(json_path) as f:
        data = json.load(f)

    cam = data.get("camera", {})
    pose = data.get("pose", {})

    return {
        "f": cam.get("f", 105790.0),
        "cx": cam.get("cx", 2048.0),
        "cy": cam.get("cy", 2048.0),
        "k1": cam.get("k1", 0.0),
        "k2": cam.get("k2", 0.0),
        "cx_rate": cam.get("cx_rate", 0.0),
        "roll": pose.get("roll", 0.0),
        "pitch": pose.get("pitch", 0.0),
        "yaw": pose.get("yaw", 0.0),
        "time_shift": pose.get("time_shift", 0.0),
        "along_rate": pose.get("along_rate", 0.0),
    }


def save_calibration(calib: Dict, json_path: str,
                     stats: Optional[Dict] = None) -> None:
    """Save calibration result to JSON."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "camera": {
            "f": calib["f"],
            "cx": calib["cx"],
            "cy": calib["cy"],
            "k1": calib.get("k1", 0.0),
            "k2": calib.get("k2", 0.0),
            "cx_rate": calib.get("cx_rate", 0.0),
        },
        "pose": {
            "time_shift": calib.get("time_shift", 0.0),
            "roll": calib.get("roll", 0.0),
            "pitch": calib.get("pitch", 0.0),
            "yaw": calib.get("yaw", 0.0),
            "along_rate": calib.get("along_rate", 0.0),
        },
    }
    if stats:
        output["stats"] = stats

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ── Image helpers ────────────────────────────────────────────────────────

def robust_histogram_stretch(img: np.ndarray) -> np.ndarray:
    """Percentile stretch (2–98 %) per channel."""
    out = np.zeros_like(img, dtype=np.uint8)
    channels = range(img.shape[-1]) if img.ndim == 3 else [None]

    for ch in channels:
        c = img[..., ch] if ch is not None else img
        valid = c[c > 0] if c.min() >= 0 else c.ravel()
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            stretched = c.astype(np.uint8)
        else:
            stretched = np.clip((c - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        if ch is not None:
            out[..., ch] = stretched
        else:
            out = stretched
    return out


def clahe_enhance(image: np.ndarray,
                  clip_limit: float = 2.0,
                  grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    cl = cv2.createCLAHE(clipLimit=clip_limit,
                         tileGridSize=(grid_size, grid_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cl.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return cl.apply(image)


def enhance_for_matching(img: np.ndarray) -> np.ndarray:
    """Full preprocessing chain for feature matching."""
    return clahe_enhance(robust_histogram_stretch(img))


# ── Satellite image loading ──────────────────────────────────────────────

def load_satellite_image(path: str) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load a satellite GeoTIFF / JP2 as RGB uint8.
    Returns (image [H, W, 3], open dataset handle).
    """
    ds = rasterio.open(path)
    if ds.count >= 3:
        data = np.transpose(ds.read([1, 2, 3]), (1, 2, 0))
    else:
        band = ds.read(1)
        data = np.stack([band, band, band], axis=-1)

    if data.dtype != np.uint8:
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return data, ds


def find_sentinel_band(sentinel_dir: str,
                       band_name: str = "TCI") -> Optional[str]:
    """Find a Sentinel-2 band file inside a directory tree.

    Supports two layouts:
    - .SAFE directory tree: searches for *<band_name>*.jp2 (e.g. *TCI*.jp2)
    - GEE-downloaded GeoTIFF: a single .tif in the directory is used directly
      when no band-name match is found (band_name is ignored in this case).
    """
    path = Path(sentinel_dir)
    valid_exts = {".jp2", ".tif", ".tiff"}

    # First: try to find a file matching the requested band name
    matches = list(path.rglob(f"*{band_name}*.jp2"))
    if not matches:
        matches = list(path.rglob(f"*{band_name}*"))
    matches = [m for m in matches if m.suffix.lower() in valid_exts]
    # Prefer files in IMG_DATA over QI_DATA (masks)
    img_data = [m for m in matches if "IMG_DATA" in str(m)]
    if img_data:
        matches = img_data
    if matches:
        return str(matches[0])

    # Fallback: single GeoTIFF (GEE download layout — no band name in filename)
    tifs = sorted(path.rglob("*.tif")) + sorted(path.rglob("*.tiff"))
    tifs = [t for t in tifs if t.suffix.lower() in valid_exts]
    return str(tifs[0]) if tifs else None


# ── Metadata helpers ─────────────────────────────────────────────────────

def load_metadata_timing(metadata_path: str) -> Optional[Dict]:
    """
    Parse PhiSat-2 session metadata JSON.
    Returns dict with keys: line_time, image_start_utc, session_key
    or None if parsing fails.
    """
    path = Path(metadata_path)
    if not path.exists():
        return None

    with open(path) as f:
        meta = json.load(f)

    # The session key is the first (and usually only) top-level key
    session_key = next(iter(meta))

    try:
        session = meta[session_key]
        line_period_us = session["ImagerConfig"]["LinePeriod"]
        time_sync = session["TimeSync"]
        exp_start_band0 = session["Scene 0"]["ExposureStart"]["Band 0"][0]

        line_time = line_period_us * 1e-6

        tick_diff = exp_start_band0 - time_sync["ImagerTime"]
        start_time_offset = tick_diff * 1e-6
        image_start_utc = time_sync["PlatformTime"] + start_time_offset

        return {
            "line_time": line_time,
            "image_start_utc": image_start_utc,
            "session_key": session_key,
        }
    except (KeyError, IndexError, TypeError) as e:
        print(f"  Warning: metadata parsing failed ({e})")
        return None
