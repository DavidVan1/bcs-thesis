"""
Robust 3-phase sensor calibration.

Phase 1 – Soft-L1 robust optimisation (tolerant to outliers)
Phase 2 – 3-σ outlier removal
Phase 3 – Final linear least-squares refinement on clean inliers

Parameter vector (9):
    [time_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .config import SceneConfig
from .sensor_model import RobustModel, create_model
from .utils import load_tie_points, save_calibration

logger = logging.getLogger(__name__)


# ── Residual function ──────────────────────────────────────────────────

def _residuals(params: np.ndarray,
               model: RobustModel,
               tie_points: List[Dict]) -> np.ndarray:
    """
    Compute [lon_err_m, lat_err_m] for every tie point.
    Returns flat array of length 2·N.
    """
    residuals = []
    for tp in tie_points:
        pred_ecef = model.predict_with_params(
            tp["phisat_x"], tp["phisat_y"], params)

        if pred_ecef is None:
            residuals.extend([1e5, 1e5])
            continue

        pred_lon, pred_lat, _ = model.ecef_to_lonlat(*pred_ecef)

        lat_res_m = (pred_lat - tp["lat"]) * 111_132.0
        lon_scale = 111_132.0 * np.cos(np.radians(tp["lat"]))
        lon_res_m = (pred_lon - tp["lon"]) * lon_scale

        residuals.append(lon_res_m)
        residuals.append(lat_res_m)

    return np.array(residuals)


# ── Calibration runner ─────────────────────────────────────────────────

def run_calibration(config: SceneConfig,
                    verbose: bool = True) -> Dict:
    """
    Full 3-phase calibration for a scene.

    Returns
    -------
    dict with keys: f, cx, cy, k1, k2, roll, pitch, yaw, time_shift, cx_rate
    Also saves the JSON to config.calib_json.
    """
    missing = config.check_inputs("calibration")
    if missing:
        raise FileNotFoundError(
            "Missing files for calibration:\n  " + "\n  ".join(missing))

    logger.info("=" * 60)
    logger.info(f"CALIBRATION — scene '{config.name}'")
    logger.info("=" * 60)

    # Model
    model = create_model(config, model_class=RobustModel)

    # Tie points
    all_points = load_tie_points(str(config.tie_points_path))
    logger.info("Loaded %d tie points.", len(all_points))

    # Bounds: [t_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate,
    #          roll_rate, pitch_rate, yaw_rate]
    x0    = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0]
    lower = [-60., -5., -5., -5., 0.8, -1e-8, -1e-8,
             -0.25, -0.03, -0.8, -0.8, -0.8]
    upper = [ 60.,  5.,  5.,  5., 1.2,  1e-8,  1e-8,
              0.25,  0.03,  0.8,  0.8,  0.8]

    # ── Phase 1: Robust (Soft L1) ──
    logger.info("\n--- Phase 1: Robust Optimisation (Soft L1) ---")
    res1 = least_squares(
        _residuals, x0, args=(model, all_points),
        bounds=(lower, upper),
        loss="soft_l1", f_scale=100.0,
        verbose=2 if verbose else 0,
    )

    # ── Phase 2: 3-σ outlier removal ──
    logger.info("\n--- Phase 2: Outlier Filtering (3-σ) ---")
    res_vec = _residuals(res1.x, model, all_points).reshape(-1, 2)
    distances = np.linalg.norm(res_vec, axis=1)

    mean_err = np.mean(distances)
    std_err = np.std(distances)
    threshold = mean_err + 3 * std_err

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
    if drift_mag > 0.5:
        logger.warning(
            f"Large attitude drift detected ({drift_mag:.3f}°). "
            f"Consider checking AOCS data quality.")

    # Warn if solution hit a bound
    for i, (val, lo, hi) in enumerate(zip(p, lower, upper)):
        if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
            names = ['time_shift','roll','pitch','yaw','f_scale',
                     'k1','k2','cx_rate','along_rate',
                     'roll_rate','pitch_rate','yaw_rate']
            logger.warning(
                f"Parameter '{names[i]}' hit bound ({val:.6f}). "
                f"Model may be under-constrained.")

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

    if config.calib_json:
        save_calibration(calib, str(config.calib_path), stats=stats)
        logger.info("Saved → %s", config.calib_path)

    return calib