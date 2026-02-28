"""
Robust 3-phase sensor calibration.

Phase 1 – Soft-L1 robust optimisation (tolerant to outliers)
Phase 2 – 3-σ outlier removal
Phase 3 – Final linear least-squares refinement on clean inliers

Parameter vector (9):
    [time_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .config import SceneConfig
from .sensor_model import RobustModel, create_model
from .utils import load_tie_points, save_calibration


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

    print("=" * 60)
    print(f"CALIBRATION — scene '{config.name}'")
    print("=" * 60)

    # Model
    model = create_model(config, model_class=RobustModel)

    # Tie points
    all_points = load_tie_points(str(config.tie_points_path))
    print(f"Loaded {len(all_points)} tie points.")

    # Bounds: [t_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate]
    x0 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    lower = [-60.0, -5.0, -5.0, -5.0, 0.8, -1e-8, -1e-8, -1.0, -0.1]
    upper = [ 60.0,  5.0,  5.0,  5.0, 1.2,  1e-8,  1e-8,  1.0,  0.1]

    # ── Phase 1: Robust (Soft L1) ──
    print("\n--- Phase 1: Robust Optimisation (Soft L1) ---")
    res1 = least_squares(
        _residuals, x0, args=(model, all_points),
        bounds=(lower, upper),
        loss="soft_l1", f_scale=100.0,
        verbose=2 if verbose else 0,
    )

    # ── Phase 2: 3-σ outlier removal ──
    print("\n--- Phase 2: Outlier Filtering (3-σ) ---")
    res_vec = _residuals(res1.x, model, all_points).reshape(-1, 2)
    distances = np.linalg.norm(res_vec, axis=1)

    mean_err = np.mean(distances)
    std_err = np.std(distances)
    threshold = mean_err + 3 * std_err

    print(f"Mean: {mean_err:.1f} m  Std: {std_err:.1f} m  "
          f"Threshold: {threshold:.1f} m")

    inliers = [tp for tp, d in zip(all_points, distances) if d < threshold]
    n_out = len(all_points) - len(inliers)
    print(f"Kept {len(inliers)} inliers.  Removed {n_out} outliers.")

    # ── Phase 3: Final refinement ──
    print("\n--- Phase 3: Final Refinement ---")
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

    print("\n" + "=" * 40)
    print("FINAL CALIBRATION RESULTS")
    print("=" * 40)
    print(f"Time Shift : {p[0]:.4f} s")
    print(f"Roll       : {p[1]:.4f}°")
    print(f"Pitch      : {p[2]:.4f}°")
    print(f"Yaw        : {p[3]:.4f}°")
    print(f"Focal Len  : {refined_f:.1f} px (scale {p[4]:.4f})")
    print(f"Distortion : k1={p[5]:.6f}  k2={p[6]:.6f}")
    print(f"CX Rate    : {p[7]:.6f} px/line")
    print(f"Along Rate : {p[8]:.6f}")
    print(f"RMSE       : {final_rmse:.1f} m")

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
    }

    stats = {
        "initial_points": len(all_points),
        "inliers": len(inliers),
        "rmse_m": final_rmse,
    }

    if config.calib_json:
        save_calibration(calib, str(config.calib_path), stats=stats)
        print(f"Saved → {config.calib_path}")

    return calib
