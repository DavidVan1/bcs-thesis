"""
Robust 3-phase sensor calibration.

Phase 1 – Soft-L1 robust optimisation (tolerant to outliers)
Phase 2 – 3-σ outlier removal
Phase 3 – Final linear least-squares refinement on clean inliers

Parameter vector (12):
    [time_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate,
     roll_rate, pitch_rate, yaw_rate]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.optimize import least_squares

from .sensor_model import RobustModel, create_model
from .utils import load_tie_points, save_calibration

logger = logging.getLogger(__name__)


# ── Calibration parameter definition ─────────────────────────────────────

PARAM_NAMES: List[str] = [
    "time_shift", "roll", "pitch", "yaw", "f_scale",
    "k1", "k2", "cx_rate", "along_rate",
    "roll_rate", "pitch_rate", "yaw_rate",
]

PARAM_INITIAL: List[float] = [
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

PARAM_LOWER: List[float] = [
    -60., -5., -5., -5., 0.8, -1e-8, -1e-8,
    -0.25, -0.03, -0.8, -0.8, -0.8,
]

PARAM_UPPER: List[float] = [
    60., 5., 5., 5., 1.2, 1e-8, 1e-8,
    0.25, 0.03, 0.8, 0.8, 0.8,
]

OUTLIER_SIGMA: float = 3.0
DRIFT_WARNING_THRESHOLD: float = 1  # degrees


def _attach_dem_heights(dem_path: Path,
                        tie_points: List[Dict]) -> List[Dict]:
    """Attach DEM elevation (metres) to each tie point as `height_m`."""
    if dem_path is None or not dem_path.exists():
        raise FileNotFoundError(f"DEM not found for calibration: {dem_path}")

    if not tie_points:
        return []

    lons = [tp["lon"] for tp in tie_points]
    lats = [tp["lat"] for tp in tie_points]

    with rasterio.open(str(dem_path)) as dem_src:
        if dem_src.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")

        to_dem = Transformer.from_crs("EPSG:4326", dem_src.crs, always_xy=True)
        xs, ys = to_dem.transform(lons, lats)
        coords = list(zip(xs, ys))

        samples = list(dem_src.sample(coords, indexes=1, masked=True))
        nodata = dem_src.nodata

    out: List[Dict] = []
    dropped = 0
    for tp, s in zip(tie_points, samples):
        val = s[0]
        if np.ma.is_masked(val):
            dropped += 1
            continue

        h = float(val)
        if not np.isfinite(h):
            dropped += 1
            continue
        if nodata is not None and h == float(nodata):
            dropped += 1
            continue

        tpe = dict(tp)
        tpe["height_m"] = h
        out.append(tpe)

    logger.info(
        "DEM enrichment: kept %d / %d tie points (dropped %d without valid DEM height)",
        len(out), len(tie_points), dropped,
    )
    return out


# ── Residual function ──────────────────────────────────────────────────

def _residuals(params: np.ndarray,
               model: RobustModel,
               tie_points: List[Dict]) -> np.ndarray:
    """
    Compute 3D ECEF residuals [dx_m, dy_m, dz_m] for each tie point.
    Returns flat array of length 3·N.
    """
    residuals: List[float] = []
    for tp in tie_points:
        pred_ecef = model.predict_with_params(
            tp["phisat_x"], tp["phisat_y"], params,
            ground_height=tp["height_m"],
        )

        if pred_ecef is None:
            residuals.extend([1e5, 1e5, 1e5])
            continue

        target_ecef = model.lonlat_to_ecef(tp["lon"], tp["lat"], tp["height_m"])
        diff = pred_ecef - target_ecef
        residuals.append(float(diff[0]))
        residuals.append(float(diff[1]))
        residuals.append(float(diff[2]))

    return np.array(residuals)


# ── Calibration runner ─────────────────────────────────────────────────

def run_calibration(
    aocs_path: Path,
    metadata_path: Optional[Path],
    tie_points_path: Path,
    dem_path: Path,
    output_path: Path,
    f: float = 105454.0,
    cx: float = 2048.0,
    cy: float = 2048.0,
    verbose: bool = True,
) -> Dict:
    """
    Full 3-phase calibration for a scene.

    Returns
    -------
    dict with keys: f, cx, cy, k1, k2, roll, pitch, yaw, time_shift, cx_rate
    Also saves the JSON to ``output_path``.
    """
    missing = []
    if not aocs_path.exists():
        missing.append(f"AOCS: {aocs_path}")
    if not tie_points_path.exists():
        missing.append(f"Tie points CSV: {tie_points_path}")
    if not dem_path.exists():
        missing.append(f"DEM: {dem_path}")
    if missing:
        raise FileNotFoundError(
            "Missing files for calibration:\n  " + "\n  ".join(missing))

    logger.info("=" * 60)
    logger.info("CALIBRATION")
    logger.info("=" * 60)

    # Model
    model = create_model(
        aocs_path,
        metadata_path,
        f=f,
        cx=cx,
        cy=cy,
        model_class=RobustModel,
    )

    # Tie points
    all_points = load_tie_points(str(tie_points_path))
    logger.info("Loaded %d tie points.", len(all_points))
    all_points = _attach_dem_heights(dem_path, all_points)
    if len(all_points) < 20:
        raise RuntimeError(
            f"Not enough DEM-enriched tie points for calibration: {len(all_points)}")

    # Bounds
    x0 = list(PARAM_INITIAL)
    lower = list(PARAM_LOWER)
    upper = list(PARAM_UPPER)

    # ── Phase 1: Robust (Soft L1) ──
    logger.info("\n--- Phase 1: Robust Optimisation (Soft L1) ---")
    res1 = least_squares(
        _residuals, x0, args=(model, all_points),
        bounds=(lower, upper),
        loss="soft_l1", f_scale=100.0,
        verbose=2 if verbose else 0,
    )

    # ── Phase 2: 3-σ outlier removal ──
    logger.info("\n--- Phase 2: Outlier Filtering (%d-σ) ---", OUTLIER_SIGMA)
    res_vec = _residuals(res1.x, model, all_points).reshape(-1, 3)
    distances = np.linalg.norm(res_vec, axis=1)

    mean_err = np.mean(distances)
    std_err = np.std(distances)
    threshold = mean_err + OUTLIER_SIGMA * std_err

    logger.info(
        "Mean: %.1f m  Std: %.1f m  Threshold: %.1f m",
        mean_err, std_err, threshold)

    inliers = [tp for tp, d in zip(all_points, distances) if d < threshold]
    n_out = len(all_points) - len(inliers)
    logger.info("Kept %d inliers.  Removed %d outliers.", len(inliers), n_out)

    # ── Phase 3: Final refinement ──
    logger.info("\n--- Phase 3: Final Refinement ---")
    res2 = least_squares(
        _residuals, res1.x, args=(model, inliers),
        bounds=(lower, upper),
        loss="linear",
        verbose=2 if verbose else 0,
        ftol=1e-6,
    )

    # ── Results ──
    p = res2.x
    refined_f = model.f * p[4]
    final_rmse = np.sqrt(np.mean(res2.fun ** 2))

    logger.info("\n" + "=" * 40)
    logger.info("FINAL CALIBRATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Time Shift : {p[0]:.4f} s")
    logger.info(f"Roll       : {p[1]:.4f}°")
    logger.info(f"Pitch      : {p[2]:.4f}°")
    logger.info(f"Yaw        : {p[3]:.4f}°")
    logger.info(f"Focal Len  : {refined_f:.1f} px (scale {p[4]:.4f})")
    logger.info(f"Distortion : k1={p[5]:.6f}  k2={p[6]:.6f}")
    logger.info(f"CX Rate    : {p[7]:.6f} px/line")
    logger.info(f"Along Rate : {p[8]:.6f}")
    logger.info(f"Roll Rate  : {p[9]:.6f} °/norm")
    logger.info(f"Pitch Rate : {p[10]:.6f} °/norm")
    logger.info(f"Yaw Rate   : {p[11]:.6f} °/norm")
    logger.info(f"RMSE       : {final_rmse:.1f} m")

    # Warn if any rate param is large — suggests real attitude drift
    drift_mag = np.hypot(p[9], np.hypot(p[10], p[11]))
    if drift_mag > DRIFT_WARNING_THRESHOLD:
        logger.warning(
            "Large attitude drift detected (%.3f°). "
            "Consider checking AOCS data quality.", drift_mag)

    # Warn if solution hit a bound
    for i, (val, lo, hi) in enumerate(zip(p, lower, upper)):
        if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
            logger.warning(
                "Parameter '%s' hit bound (%.6f). "
                "Model may be under-constrained.",
                PARAM_NAMES[i], val)

    calib = {
        "f": refined_f,
        "cx": model.cx,
        "cy": model.cy,
        "k1": p[5],
        "k2": p[6],
        "time_shift": p[0],
        "roll": p[1],
        "pitch": p[2],
        "yaw": p[3],
        "cx_rate": p[7],
        "along_rate": p[8],
        "roll_rate": p[9],
        "pitch_rate": p[10],
        "yaw_rate": p[11],
    }

    stats = {
        "initial_points": len(all_points),
        "inliers": len(inliers),
        "rmse_m": final_rmse,
    }

    save_calibration(calib, str(output_path), stats=stats)
    logger.info("Saved → %s", output_path)

    return calib