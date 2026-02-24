"""
DEM-aware robust sensor calibration.

Five-phase pipeline
===================
Phase 1 – Coarse pose-only fit (Cauchy, large f_scale for loss)
Phase 2 – Full 10-parameter Cauchy fit
Phase 3 – 3-σ outlier removal
Phase 4 – Refined Cauchy fit on inliers
Phase 5 – Final linear least-squares polish

Parameter vector (10):
    [time_shift, roll, pitch, yaw, f_scale, k1, k2,
     dcx, along_scale, dcy]

Key improvements over the previous 7-param / sphere calibration:
- WGS-84 ellipsoid + iterative DEM ray intersection
- Principal-point offsets  dcx, dcy  (pixels)
- Along-track timing scale  along_scale  (multiplicative on line_time)
- Cauchy loss (heavier tails than soft-L1)
- Proper parameter scaling via ``x_scale``
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from scipy.optimize import least_squares

from .config import SceneConfig
from .sensor_model import RobustModel, create_model
from .utils import load_tie_points, save_calibration


# ═══════════════════════════════════════════════════════════════════════════
# DEM loader (used only during calibration)
# ═══════════════════════════════════════════════════════════════════════════

class _DEMInterpolator:
    """Bilinear DEM query for calibration.  Kept alive for the run."""

    def __init__(self, dem_path: str):
        self._src = rasterio.open(dem_path)
        self._data = self._src.read(1).astype(np.float64)
        self._tf = self._src.transform

    def __call__(self, lon: float, lat: float) -> Optional[float]:
        px = (lon - self._tf.c) / self._tf.a
        py = (lat - self._tf.f) / self._tf.e
        if px < 0 or px >= self._data.shape[1] or py < 0 or py >= self._data.shape[0]:
            return None
        ix, iy = int(np.floor(px)), int(np.floor(py))
        if ix + 1 >= self._data.shape[1] or iy + 1 >= self._data.shape[0]:
            return None
        fx, fy = px - ix, py - iy
        z = ((1 - fx) * (1 - fy) * self._data[iy, ix]
             + fx * (1 - fy) * self._data[iy, ix + 1]
             + (1 - fx) * fy * self._data[iy + 1, ix]
             + fx * fy * self._data[iy + 1, ix + 1])
        return float(z)

    def close(self):
        self._src.close()


# ═══════════════════════════════════════════════════════════════════════════
# Residual function
# ═══════════════════════════════════════════════════════════════════════════

def _residuals(params: np.ndarray,
               model: RobustModel,
               tie_points: List[Dict]) -> np.ndarray:
    """
    Compute [easting_err_m, northing_err_m] for every tie point.
    Returns flat array of length 2·N.
    """
    residuals = np.empty(2 * len(tie_points), dtype=np.float64)

    for i, tp in enumerate(tie_points):
        pred_ecef = model.predict_with_params(
            tp["phisat_x"], tp["phisat_y"], params)

        if pred_ecef is None:
            residuals[2 * i]     = 1e5
            residuals[2 * i + 1] = 1e5
            continue

        pred_lon, pred_lat, _ = model.ecef_to_lonlat(*pred_ecef)

        lat_res_m = (pred_lat - tp["lat"]) * 111_132.0
        lon_scale = 111_132.0 * np.cos(np.radians(tp["lat"]))
        lon_res_m = (pred_lon - tp["lon"]) * lon_scale

        residuals[2 * i]     = lon_res_m
        residuals[2 * i + 1] = lat_res_m

    return residuals


# ═══════════════════════════════════════════════════════════════════════════
# Calibration runner
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration(config: SceneConfig,
                    verbose: bool = True) -> Dict:
    """
    Five-phase DEM-aware calibration for a scene.

    Returns
    -------
    dict with keys:
        f, cx, cy, k1, k2, roll, pitch, yaw, time_shift, along_scale
    Also saves the JSON to config.calib_json.
    """
    missing = config.check_inputs("calibration")
    if missing:
        raise FileNotFoundError(
            "Missing files for calibration:\n  " + "\n  ".join(missing))

    print("=" * 60)
    print(f"CALIBRATION — scene '{config.name}'")
    print("=" * 60)

    # ── Model + DEM ──
    model = create_model(config, model_class=RobustModel)

    dem = None
    if config.dem_path and config.dem_path.exists():
        dem = _DEMInterpolator(str(config.dem_path))
        model.set_dem(dem)
        print(f"DEM attached for calibration: {config.dem_path}")
    else:
        print("WARNING: No DEM — calibrating against ellipsoid only.")

    # ── Tie points ──
    all_points = load_tie_points(str(config.tie_points_path))
    print(f"Loaded {len(all_points)} tie points.")

    # ─────────────────────────────────────────────────────────────────
    # Parameter layout (10):
    #   0: time_shift  (s)        5: k1
    #   1: roll        (°)        6: k2
    #   2: pitch       (°)        7: dcx    (px)
    #   3: yaw         (°)        8: along_scale
    #   4: f_scale     (×)        9: dcy    (px)
    # ─────────────────────────────────────────────────────────────────
    #                t_sh  roll  pitch yaw   fsc     k1      k2     dcx  ascl    dcy
    x0    = np.array([0.0,  0.0,  0.0, 0.0,  1.0,    0.0,    0.0,   0.0, 1.0,    0.0])
    lower = np.array([-60., -5.,  -5., -5.,  0.8,   -1e-6,  -1e-12, -50., 0.9,  -50.])
    upper = np.array([ 60.,  5.,   5.,  5.,  1.2,    1e-6,   1e-12,  50., 1.1,   50.])

    # x_scale: characteristic step for each parameter (for LM Jacobian scaling)
    x_scale = np.array([1.0, 0.01, 0.01, 0.01, 0.001, 1e-8, 1e-14, 1.0, 0.001, 1.0])

    verb = 2 if verbose else 0

    # ── Phase 1: Coarse pose-only (freeze intrinsics + cx/cy) ──────
    print("\n--- Phase 1: Coarse Pose Fit (Cauchy) ---")
    # Only optimise time_shift, roll, pitch, yaw; clamp others at init
    # Use a tiny epsilon band around frozen values (scipy requires lb < ub strictly)
    _eps = 1e-15
    lower1 = x0 - _eps;  upper1 = x0 + _eps
    lower1[:4] = lower[:4];  upper1[:4] = upper[:4]
    # Open f_scale slightly
    lower1[4] = 0.95;  upper1[4] = 1.05

    res1 = least_squares(
        _residuals, x0, args=(model, all_points),
        bounds=(lower1, upper1),
        loss="cauchy", f_scale=50.0,
        x_scale=x_scale,
        method="trf", verbose=verb,
        max_nfev=300,
    )
    _print_params("Phase 1", res1.x, model)

    # ── Phase 2: Full 10-parameter fit ─────────────────────────────
    print("\n--- Phase 2: Full 10-Parameter Fit (Cauchy) ---")
    res2 = least_squares(
        _residuals, res1.x, args=(model, all_points),
        bounds=(lower, upper),
        loss="cauchy", f_scale=30.0,
        x_scale=x_scale,
        method="trf", verbose=verb,
        max_nfev=500,
    )
    _print_params("Phase 2", res2.x, model)

    # ── Phase 3: 3-σ outlier removal ───────────────────────────────
    print("\n--- Phase 3: Outlier Filtering (3-σ) ---")
    res_vec = _residuals(res2.x, model, all_points).reshape(-1, 2)
    distances = np.linalg.norm(res_vec, axis=1)

    med = np.median(distances)
    mad = np.median(np.abs(distances - med)) * 1.4826   # robust σ
    threshold = med + 3.0 * mad

    print(f"  Median error: {med:.1f} m   MAD-σ: {mad:.1f} m   "
          f"Threshold: {threshold:.1f} m")

    inliers = [tp for tp, d in zip(all_points, distances) if d < threshold]
    n_out = len(all_points) - len(inliers)
    print(f"  Kept {len(inliers)} inliers.  Removed {n_out} outliers.")

    # ── Phase 4: Re-fit with Cauchy on inliers ─────────────────────
    print("\n--- Phase 4: Cauchy Refinement on Inliers ---")
    res4 = least_squares(
        _residuals, res2.x, args=(model, inliers),
        bounds=(lower, upper),
        loss="cauchy", f_scale=15.0,
        x_scale=x_scale,
        method="trf", verbose=verb,
        max_nfev=500,
    )
    _print_params("Phase 4", res4.x, model)

    # ── Phase 5: Final linear polish ───────────────────────────────
    print("\n--- Phase 5: Final Linear Polish ---")
    res5 = least_squares(
        _residuals, res4.x, args=(model, inliers),
        bounds=(lower, upper),
        loss="linear",
        x_scale=x_scale,
        method="trf", verbose=verb,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=1000,
    )
    _print_params("Phase 5 (FINAL)", res5.x, model)

    # ── Build output ───────────────────────────────────────────────
    p = res5.x
    refined_f  = model.f * p[4]
    refined_cx = model.cx + p[7]
    refined_cy = model.cy + p[9]
    final_rmse = np.sqrt(np.mean(res5.fun ** 2))

    print(f"\n  FINAL RMSE: {final_rmse:.2f} m  "
          f"({final_rmse / 4.75:.2f} px @ 4.75 m GSD)")

    calib = {
        "f": refined_f,
        "cx": refined_cx,
        "cy": refined_cy,
        "k1": p[5],
        "k2": p[6],
        "time_shift": p[0],
        "roll": p[1],
        "pitch": p[2],
        "yaw": p[3],
        "along_scale": p[8],
    }

    stats = {
        "initial_points": len(all_points),
        "inliers": len(inliers),
        "outliers_removed": n_out,
        "rmse_m": final_rmse,
        "rmse_px": final_rmse / 4.75,
        "phases": {
            "phase1_cost": float(res1.cost),
            "phase2_cost": float(res2.cost),
            "phase4_cost": float(res4.cost),
            "phase5_cost": float(res5.cost),
        },
    }

    if config.calib_json:
        save_calibration(calib, str(config.calib_path), stats=stats)
        print(f"Saved → {config.calib_path}")

    if dem is not None:
        dem.close()

    return calib


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_params(label: str, p: np.ndarray, model: RobustModel) -> None:
    rmse = np.sqrt(np.mean(
        _residuals(p, model, []) ** 2)) if False else None  # skip if no pts
    refined_f = model.f * p[4]
    print(f"\n  ── {label} ──")
    print(f"  Time Shift   : {p[0]:+.4f} s")
    print(f"  Roll         : {p[1]:+.5f}°")
    print(f"  Pitch        : {p[2]:+.5f}°")
    print(f"  Yaw          : {p[3]:+.5f}°")
    print(f"  Focal Length : {refined_f:.1f} px  (scale {p[4]:.6f})")
    print(f"  Distortion   : k1={p[5]:.2e}  k2={p[6]:.2e}")
    print(f"  Principal Pt : dcx={p[7]:+.2f} px   dcy={p[9]:+.2f} px")
    print(f"  Along Scale  : {p[8]:.6f}")
