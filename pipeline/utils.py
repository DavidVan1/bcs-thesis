"""
Shared utilities: image I/O, enhancement, tie-point loading.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import cv2
import rasterio

from .config import DEFAULT_FOCAL_LENGTH, PROJECT_ROOT

logger = logging.getLogger(__name__)


def configure_pipeline_file_logger(log_path: Optional[Path] = None) -> logging.Logger:
    """Configure and return a dedicated logger writing compact stage lines to pipeline.log."""
    ts = datetime.now().strftime("%Y%m%d")
    target = log_path or (PROJECT_ROOT / f"logs/pipeline_{ts}.log")
    target.parent.mkdir(parents=True, exist_ok=True)

    pipeline_logger = logging.getLogger("phisat_pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.propagate = False

    target_resolved = target.resolve()
    for handler in pipeline_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", "")).resolve()
            if existing == target_resolved:
                return pipeline_logger

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    pipeline_logger.addHandler(file_handler)
    return pipeline_logger


def log_pipeline_stage(
    *,
    scene: str,
    matcher: str,
    stage: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    """Write a single structured line for a stage outcome to phisat_pipeline logger."""
    line_parts = [
        f"scene={scene}",
        f"matcher={matcher}",
        f"stage={stage}",
        f"status={status}",
    ]

    if metrics:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float):
                line_parts.append(f"{key}={value:.3f}")
            else:
                line_parts.append(f"{key}={value}")

    if reason:
        safe_reason = "_".join(str(reason).strip().split())[:240]
        line_parts.append(f"reason={safe_reason}")

    configure_pipeline_file_logger().info(" ".join(line_parts))


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



def load_calibration(json_path: str) -> Dict:
    """
    Load calibration JSON and return a flat dict of runtime parameters
    used by the RPC fitting pipeline.
    """
    with open(json_path) as f:
        data = json.load(f)

    cam = data.get("camera", {})
    pose = data.get("pose", {})

    out = {
        "f": cam.get("f", DEFAULT_FOCAL_LENGTH),
        "k1": cam.get("k1", 0.0),
        "k2": cam.get("k2", 0.0),
        "f_scale": cam.get("f_scale", 1.0),
        "cx_bias": cam.get("cx_bias", 0.0),
        "cy_bias": cam.get("cy_bias", 0.0),
        "cx_rate": cam.get("cx_rate", 0.0),
        "boresight_roll": pose.get("boresight_roll", pose.get("roll", 0.0)),
        "boresight_pitch": pose.get("boresight_pitch", pose.get("pitch", 0.0)),
        "boresight_yaw": pose.get("boresight_yaw", pose.get("yaw", 0.0)),
        "time_shift": pose.get("time_shift", 0.0),
        "along_rate": pose.get("along_rate", 0.0),
        "drift_roll_1": pose.get("drift_roll_1", pose.get("roll_rate", 0.0)),
        "drift_pitch_1": pose.get("drift_pitch_1", pose.get("pitch_rate", 0.0)),
        "drift_yaw_1": pose.get("drift_yaw_1", pose.get("yaw_rate", 0.0)),
    }
    return out


def save_calibration(calib: Dict, json_path: str,
                     stats: Optional[Dict] = None) -> None:
    """Save calibration result to JSON."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    camera = {
        "f": calib["f"],
        "cx": calib["cx"],
        "cy": calib["cy"],
    }
    for key in ["f_scale", "cx_bias", "cy_bias", "k1", "k2", "cx_rate"]:
        if key in calib:
            camera[key] = calib[key]

    pose = {}
    for key in [
        "time_shift",
        "roll", "pitch", "yaw",
        "boresight_roll", "boresight_pitch", "boresight_yaw",
        "along_rate",
        "roll_rate", "pitch_rate", "yaw_rate",
        "drift_roll_1", "drift_pitch_1", "drift_yaw_1",
    ]:
        if key in calib:
            pose[key] = calib[key]

    output = {
        "camera": camera,
        "pose": pose,
    }
    if "model_v2" in calib:
        output["model_v2"] = calib["model_v2"]
    if stats:
        output["stats"] = stats

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ── Image helpers ────────────────────────────────────────────────────────

def robust_histogram_stretch(img: np.ndarray, high_percentile=98) -> np.ndarray:
    """Percentile stretch (2–98 %) per channel."""
    out = np.zeros_like(img, dtype=np.uint8)
    channels = range(img.shape[-1]) if img.ndim == 3 else [None]

    for ch in channels:
        c = img[..., ch] if ch is not None else img
        valid = c[c > 0] if c.min() >= 0 else c.ravel()
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, (3, high_percentile))
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


def enhance_for_matching(img: np.ndarray, high_percentile=98) -> np.ndarray:
    """Full preprocessing chain for feature matching."""
    return clahe_enhance(robust_histogram_stretch(img, high_percentile))


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
        logger.warning("Metadata parsing failed: %s", e)
        return None