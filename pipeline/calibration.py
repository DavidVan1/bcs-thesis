"""
Robust 3-phase sensor calibration.

Phase 1 – Soft-L1 robust optimisation (tolerant to outliers)
Phase 2 – 3-σ outlier removal
Phase 3 – Final linear least-squares refinement on clean inliers
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.optimize import least_squares

from .sensor_model import RobustModel, create_model
from .utils import load_tie_points, save_calibration

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParamSpec:
    """Single optimization parameter definition for camera model."""
    name: str
    initial: float
    lower: float
    upper: float


def _get_default_param_specs() -> Dict[str, ParamSpec]:
    """Centralized conservative bounds/initials."""
    specs = [
        ParamSpec("time_shift", 0.0, -60.0, 60.0),
        ParamSpec("boresight_roll", 0.0, -5.0, 5.0),
        ParamSpec("boresight_pitch", 0.0, -5.0, 5.0),
        ParamSpec("boresight_yaw", 0.0, -5.0, 5.0),
        ParamSpec("f_scale", 1.0, 0.9, 1.1),
        ParamSpec("cx_bias", 0.0, -20.0, 20.0),
        ParamSpec("cy_bias", 0.0, -20.0, 20.0),
        ParamSpec("k1", 0.0, -1e-8, 1e-8),
        ParamSpec("k2", 0.0, -1e-8, 1e-8),
        ParamSpec("drift_roll_1", 0.0, -0.8, 0.8),
        ParamSpec("drift_pitch_1", 0.0, -0.8, 0.8),
        ParamSpec("drift_yaw_1", 0.0, -0.8, 0.8),
        ParamSpec("cx_rate", 0.0, -0.25, 0.25),
        ParamSpec("along_rate", 0.0, -0.03, 0.03),
    ]
    return {s.name: s for s in specs}


@dataclass
class CalibrationConfig:
    """Configuration for the calibration optimization process."""
    params: Dict[str, ParamSpec] = field(default_factory=_get_default_param_specs)
    outlier_sigma: float = 3.0
    drift_warning_threshold: float = 1.0  # degrees
    robust_loss_scale: float = 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

class DataEnricher:
    """Handles spatial data preparation prior to calibration."""

    @staticmethod
    def attach_dem_heights(dem_path: Path, tie_points: List[Dict]) -> List[Dict]:
        """Attach DEM elevation (metres) to each tie point as `height_m`."""
        if not dem_path.exists():
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
            
            samples = list(dem_src.sample(list(zip(xs, ys)), indexes=1, masked=True))
            nodata = dem_src.nodata

        out_points = []
        dropped = 0
        
        for tp, sample in zip(tie_points, samples):
            val = sample[0]
            if np.ma.is_masked(val) or not np.isfinite(float(val)) or (nodata is not None and float(val) == float(nodata)):
                dropped += 1
                continue

            tpe = dict(tp)
            tpe["height_m"] = float(val)
            out_points.append(tpe)

        logger.info("DEM enrichment: kept %d / %d tie points (dropped %d)", len(out_points), len(tie_points), dropped)
        return out_points


# ═══════════════════════════════════════════════════════════════════════════
# Core Optimization Logic
# ═══════════════════════════════════════════════════════════════════════════

class CameraCalibrator:
    """Stateful handler for the 3-phase robust calibration process."""

    def __init__(self, model: RobustModel, config: Optional[CalibrationConfig] = None):
        self.model = model
        self.config = config or CalibrationConfig()
        self.param_names = list(self.config.params.keys())

    @property
    def bounds(self) -> Tuple[List[float], List[float]]:
        lower = [s.lower for s in self.config.params.values()]
        upper = [s.upper for s in self.config.params.values()]
        return lower, upper

    @property
    def initial_guess(self) -> np.ndarray:
        return np.array([s.initial for s in self.config.params.values()])

    def unpack_params(self, values: np.ndarray) -> Dict[str, float]:
        """Convert an optimization vector into a named dictionary."""
        out = {name: spec.initial for name, spec in self.config.params.items()}
        for i, name in enumerate(self.param_names):
            out[name] = float(values[i])
        return out

    def _residuals(self, params: np.ndarray, tie_points: List[Dict]) -> np.ndarray:
        """Compute 3D ECEF residuals [dx_m, dy_m, dz_m] for each tie point."""
        residuals = []
        for tp in tie_points:
            pred_ecef = self.model.predict_with_params(
                tp["phisat_x"], tp["phisat_y"], params,
                ground_height=tp["height_m"],
                param_names=self.param_names,
            )

            if pred_ecef is None:
                residuals.extend([1e5, 1e5, 1e5])
                continue

            target_ecef = self.model.lonlat_to_ecef(tp["lon"], tp["lat"], tp["height_m"])
            diff = pred_ecef - target_ecef
            residuals.extend([float(diff[0]), float(diff[1]), float(diff[2])])

        return np.array(residuals)

    def _filter_outliers(self, params: np.ndarray, tie_points: List[Dict]) -> Tuple[List[Dict], float, float]:
        """Phase 2: Evaluate residuals and remove points beyond N-sigma."""
        res_vec = self._residuals(params, tie_points).reshape(-1, 3)
        distances = np.linalg.norm(res_vec, axis=1)

        mean_err = np.mean(distances)
        std_err = np.std(distances)
        threshold = mean_err + self.config.outlier_sigma * std_err

        inliers = [tp for tp, d in zip(tie_points, distances) if d < threshold]
        return inliers, mean_err, threshold

    def calibrate(self, tie_points: List[Dict], verbose: bool = True) -> Tuple[Dict, Dict, least_squares]:
        """Execute the full 3-phase calibration."""
        if len(tie_points) < 20:
            raise RuntimeError(f"Insufficient tie points for calibration: {len(tie_points)}")

        opt_kwargs = {"bounds": self.bounds, "verbose": 2 if verbose else 0}

        # ── Phase 1: Robust (Soft L1) ──
        logger.info("\n--- Phase 1: Robust Optimisation (Soft L1) ---")
        res1 = least_squares(
            self._residuals, self.initial_guess, args=(tie_points,),
            loss="soft_l1", f_scale=self.config.robust_loss_scale, **opt_kwargs
        )

        # ── Phase 2: Outlier Filtering ──
        logger.info("\n--- Phase 2: Outlier Filtering (%.1f-σ) ---", self.config.outlier_sigma)
        inliers, mean_err, threshold = self._filter_outliers(res1.x, tie_points)
        logger.info("Mean Err: %.1f m | Threshold: %.1f m", mean_err, threshold)
        logger.info("Kept %d inliers. Removed %d outliers.", len(inliers), len(tie_points) - len(inliers))

        # ── Phase 3: Final refinement ──
        logger.info("\n--- Phase 3: Final Refinement ---")
        res_final = least_squares(
            self._residuals, res1.x, args=(inliers,),
            loss="linear", ftol=1e-6, **opt_kwargs
        )

        # ── Evaluate Results ──
        p_opt = self.unpack_params(res_final.x)
        final_rmse = np.sqrt(np.mean(res_final.fun ** 2))
        
        stats = {
            "initial_points": len(tie_points),
            "inliers": len(inliers),
            "rmse_m": final_rmse,
            "model_v2": {"active_parameter_count": len(self.param_names)},
        }

        self._log_and_warn_results(res_final, p_opt, final_rmse)

        return p_opt, stats, res_final

    def _log_and_warn_results(self, res: least_squares, p_opt: Dict[str, float], rmse: float):
        """Log final optimized parameters and warn on boundaries or severe drift."""
        refined_f = self.model.f * p_opt["f_scale"]
        
        logger.info("\n" + "=" * 40)
        logger.info("FINAL CALIBRATION RESULTS")
        logger.info("=" * 40)
        for name in self.param_names:
            logger.info("  %-16s: %.6f", name, p_opt[name])
        logger.info("Focal Len       : %.1f px (scale %.4f)", refined_f, p_opt['f_scale'])
        logger.info("RMSE            : %.1f m", rmse)

        # Warnings
        drift_mag = np.hypot(p_opt["drift_roll_1"], np.hypot(p_opt["drift_pitch_1"], p_opt["drift_yaw_1"]))
        if drift_mag > self.config.drift_warning_threshold:
            logger.warning("Large attitude drift detected (%.3f°). Check AOCS data quality.", drift_mag)

        lower, upper = self.bounds
        for i, (val, lo, hi) in enumerate(zip(res.x, lower, upper)):
            if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
                logger.warning("Parameter '%s' hit bound (%.6f). Model may be under-constrained.", self.param_names[i], val)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Runner (Backward Compatible API)
# ═══════════════════════════════════════════════════════════════════════════

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
    Acts as the main entry point, combining I/O, models, and optimization.
    """
    # 1. File Validation
    missing = [p for p in (aocs_path, tie_points_path, dem_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for calibration: {', '.join(map(str, missing))}")

    logger.info("=" * 60)
    logger.info("CALIBRATION")
    logger.info("=" * 60)

    # 2. Setup Model & Data
    model = create_model(aocs_path, metadata_path, f=f, cx=cx, cy=cy, model_class=RobustModel)
    
    raw_points = load_tie_points(str(tie_points_path))
    enriched_points = DataEnricher.attach_dem_heights(dem_path, raw_points)

    # 3. Execute Calibration
    calibrator = CameraCalibrator(model)
    p_opt, stats, _ = calibrator.calibrate(enriched_points, verbose=verbose)

    # 4. Format & Save Output
    calib = {
        "f": model.f * p_opt["f_scale"],
        "cx": model.cx,
        "cy": model.cy,
        **p_opt,
        "model_v2": {
            "active_parameter_names": calibrator.param_names,
            "recommended_default_profile": "camera_model_v2.1",
        }
    }

    save_calibration(calib, str(output_path), stats=stats)
    logger.info("Saved → %s", output_path)

    return calib