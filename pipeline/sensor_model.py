import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation as SciRot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeodeticContext:
    """Geodetic and atmospheric constants (defaults to WGS84 / Earth)."""
    a_axis: float = 6_378_137.0
    b_axis: float = 6_356_752.314245
    atmos_beta: float = 2.0e-5
    atmos_scale_height: float = 8_000.0


@dataclass
class OptimizationParams:
    """Container for calibration parameters."""
    time_shift: float = 0.0
    boresight_roll: float = 0.0
    boresight_pitch: float = 0.0
    boresight_yaw: float = 0.0
    f_scale: float = 1.0
    cx_bias: float = 0.0
    cy_bias: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    drift_roll_1: float = 0.0
    drift_pitch_1: float = 0.0
    drift_yaw_1: float = 0.0
    cx_rate: float = 0.0
    along_rate: float = 0.0

    @classmethod
    def from_mixed_input(
        cls, 
        params: Union[np.ndarray, Dict[str, float]], 
        param_names: Optional[List[str]] = None
    ) -> "OptimizationParams":
        """Factory to safely parse arrays, dicts, and legacy aliases into a typed object."""
        inst = cls()

        # Handle Dict Input
        if isinstance(params, dict):
            alias_map = {
                "roll": "boresight_roll", "pitch": "boresight_pitch", "yaw": "boresight_yaw",
                "roll_rate": "drift_roll_1", "pitch_rate": "drift_pitch_1", "yaw_rate": "drift_yaw_1",
            }
            for key, val in params.items():
                target_key = alias_map.get(key, key)
                if hasattr(inst, target_key):
                    setattr(inst, target_key, float(val))
            return inst

        # Handle Array Input with Names
        arr = np.asarray(params, dtype=np.float64)
        if param_names is not None:
            for idx, name in enumerate(param_names[:len(arr)]):
                if hasattr(inst, name):
                    setattr(inst, name, float(arr[idx]))
            return inst

        # Legacy Vector Fallback (Unnamed Array)
        def _get(idx: int, default: float = 0.0) -> float:
            return float(arr[idx]) if len(arr) > idx else default

        inst.time_shift = _get(0)
        inst.boresight_roll = _get(1)
        inst.boresight_pitch = _get(2)
        inst.boresight_yaw = _get(3)
        inst.f_scale = _get(4, 1.0)
        inst.k1 = _get(5)
        inst.k2 = _get(6)
        inst.cx_rate = _get(7)
        inst.along_rate = _get(8)
        inst.drift_roll_1 = _get(9)
        inst.drift_pitch_1 = _get(10)
        inst.drift_yaw_1 = _get(11)
        return inst


class PhiSatPushbroomModel:
    """Core geometric model for a pushbroom line scanner."""

    _GEODETIC_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    _ECEF_TO_GEODETIC = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

    def __init__(self, f: float = 3000.0, cx: float = 2048.0, cy: float = 2048.0,
                 k1: float = 0.0, k2: float = 0.0, 
                 geo_context: Optional[GeodeticContext] = None):
        self.f = f
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.geo = geo_context or GeodeticContext()

        # Platform state
        self.position: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.quaternion: Optional[np.ndarray] = None
        self.t0: Optional[float] = None

        # Timing model
        self.row0 = cy
        self.line_time = 0.001
        self.along_scale = 1.0
        self.along_shift = 0.0

        # Drift boundaries
        self.row_min: Optional[float] = None
        self.row_max: Optional[float] = None
        self._row_range_warning_emitted = False

    # ── Metadata Loading ────────────────────────────────────────────

    def load_aocs_metadata(self, json_path: str, acquisition_index: int = 0) -> None:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"AOCS file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        acq = data["Acquisitions"][acquisition_index]
        self.position = np.array([acq["OBCPositionX"], acq["OBCPositionY"], acq["OBCPositionZ"]])
        self.velocity = np.array([acq["OBCVelocityX"], acq["OBCVelocityY"], acq["OBCVelocityZ"]])
        self.quaternion = np.array(acq["QPointing"])
        self.t0 = acq["ADCSTimeSec"] + acq["ADCSTimeNs"] * 1e-9

    def set_row_range(self, row_min: float, row_max: float) -> None:
        if not np.isfinite(row_min) or not np.isfinite(row_max) or row_max <= row_min:
            logger.warning("Invalid row range provided: [%.3f, %.3f]", row_min, row_max)
            return
        self.row_min = float(row_min)
        self.row_max = float(row_max)
        logger.info("Using row range for normalized line time: [%.1f, %.1f]", self.row_min, self.row_max)

    # ── Geometry Mathematics ────────────────────────────────────────

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    @staticmethod
    def get_orbital_rotation_matrix(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        z_orb = -pos / np.linalg.norm(pos)
        y_orb = -np.cross(pos, vel)
        y_orb /= np.linalg.norm(y_orb)
        x_orb = np.cross(y_orb, z_orb)
        x_orb /= np.linalg.norm(x_orb)
        return np.column_stack([x_orb, y_orb, z_orb])

    # ── Projection Helpers ──────────────────────────────────────────

    def scanline_to_time(self, row: float) -> float:
        dt = ((row - self.row0) * self.line_time * self.along_scale + self.along_shift)
        return self.t0 + dt

    @staticmethod
    def ecef_to_lonlat(x: float, y: float, z: float) -> Tuple[float, float, float]:
        lon, lat, h = PhiSatPushbroomModel._ECEF_TO_GEODETIC.transform(float(x), float(y), float(z))
        return float(lon), float(lat), float(h)

    @staticmethod
    def lonlat_to_ecef(lon: float, lat: float, height: float = 0.0) -> np.ndarray:
        x, y, z = PhiSatPushbroomModel._GEODETIC_TO_ECEF.transform(float(lon), float(lat), float(height))
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    def predict_ground_coordinates(self, px: float, py: float, ground_height: float = 0.0) -> Optional[Tuple[float, float]]:
        ecef = self.pixel_to_ground(px, py, ground_height=ground_height)
        if ecef is None:
            return None
        lon, lat, _ = self.ecef_to_lonlat(*ecef)
        return lon, lat

    def pixel_to_ground(self, px: float, py: float, ground_height: float = 0.0) -> Optional[np.ndarray]:
        """Base implementation of pixel-to-ground without optimization parameters."""
        if self.position is None:
            raise ValueError("Must load AOCS metadata first")

        # Camera Ray
        x_norm = (px - self.cx) / self.f
        r2 = x_norm**2
        radial = 1.0 + self.k1 * r2 + self.k2 * r2**2
        ray_cam = np.array([0.0, x_norm * radial, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Platform Pose
        t = self.scanline_to_time(py)
        sat_pos = self.position + self.velocity * (t - self.t0)
        
        # Rotations
        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)

        # Intersect
        ray_world = R_o2e @ R_b2o @ ray_cam
        ray_world = _apply_atmospheric_refraction(ray_world, sat_pos, float(ground_height), self.geo)
        return _intersect_ellipsoid(sat_pos, ray_world, float(ground_height), self.geo)


class RobustModel(PhiSatPushbroomModel):
    """Pushbroom model with camera model parameters used by calibration."""

    def normalized_line_time(self, py: float) -> float:
        """Calculate normalized time [-1, 1] for drift calculations."""
        py_f = float(py)
        if self.row_min is not None and self.row_max is not None and self.row_max > self.row_min:
            t = 2.0 * (py_f - self.row_min) / (self.row_max - self.row_min) - 1.0
            return float(np.clip(t, -1.0, 1.0))

        if not self._row_range_warning_emitted:
            logger.warning("Using legacy cy-based fallback for normalized line time.")
            self._row_range_warning_emitted = True

        denom = max(float(2.0 * max(self.cy, 1.0)), 1.0)
        t = (py_f - float(self.cy)) / denom
        return float(np.clip(t, -1.0, 1.0))

    def _build_camera_ray(self, px: float, t_norm: float, p: OptimizationParams) -> Optional[np.ndarray]:
        """Atomic Step 1: Construct the internal camera ray."""
        current_f = self.f * p.f_scale
        if current_f <= 0.0:
            return None

        cx_eff = self.cx + p.cx_bias + (p.cx_rate * t_norm)
        x_norm = (float(px) - cx_eff) / current_f
        y_norm = -float(p.cy_bias) / current_f

        if p.k1 != 0.0 or p.k2 != 0.0:
            r2 = x_norm**2 + y_norm**2
            radial = 1.0 + p.k1 * r2 + p.k2 * r2**2
            xd, yd = x_norm * radial, y_norm * radial
        else:
            xd, yd = x_norm, y_norm

        ray_cam = np.array([0.0, xd, 1.0], dtype=np.float64)
        return ray_cam / np.linalg.norm(ray_cam)

    def _apply_body_rotations(self, ray_cam: np.ndarray, t_norm: float, p: OptimizationParams) -> np.ndarray:
        """Atomic Step 2: Apply static boresight and dynamic drift."""
        roll_eff = p.boresight_roll + (p.drift_roll_1 * t_norm)
        pitch_eff = p.boresight_pitch + (p.drift_pitch_1 * t_norm)
        yaw_eff = p.boresight_yaw + (p.drift_yaw_1 * t_norm)
        
        rot = SciRot.from_euler("xyz", [roll_eff, pitch_eff, yaw_eff], degrees=True)
        return rot.apply(ray_cam)

    def _calculate_platform_pose(self, py: float, p: OptimizationParams) -> Tuple[np.ndarray, np.ndarray]:
        """Atomic Step 3: Compute position and construct world rotation matrices."""
        t = self.scanline_to_time(float(py)) + p.time_shift
        dt = t - self.t0
        sat_pos = self.position + self.velocity * dt * (1.0 + p.along_rate)

        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)
        
        return sat_pos, R_o2e @ R_b2o

    def predict_with_params(self, px: float, py: float,
                            params: Union[np.ndarray, Dict[str, float]],
                            ground_height: float = 0.0,
                            param_names: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Pipeline orchestrator for calibrated forward-projection."""
        # 1. Parse Parameters safely
        p = OptimizationParams.from_mixed_input(params, param_names)
        t_norm = self.normalized_line_time(py)

        # 2. Build Ray in Camera Space
        ray_cam = self._build_camera_ray(px, t_norm, p)
        if ray_cam is None:
            return None

        # 3. Apply Local Boresight & Drift
        ray_body = self._apply_body_rotations(ray_cam, t_norm, p)

        # 4. Resolve World Pose
        sat_pos, R_world = self._calculate_platform_pose(py, p)
        ray_world = R_world @ ray_body

        # 5. Environment Refraction & Intersection
        ray_world = _apply_atmospheric_refraction(ray_world, sat_pos, float(ground_height), self.geo)
        return _intersect_ellipsoid(sat_pos, ray_world, float(ground_height), self.geo)


def _intersect_ellipsoid(origin: np.ndarray, direction: np.ndarray, 
                         ground_height: float, geo: GeodeticContext) -> Optional[np.ndarray]:
    """Ray–WGS84-ellipsoid intersection at constant ellipsoidal height."""
    a_axis = geo.a_axis + float(ground_height)
    b_axis = geo.b_axis + float(ground_height)

    ox, oy, oz = origin
    dx, dy, dz = direction

    inv_a2 = 1.0 / (a_axis * a_axis)
    inv_b2 = 1.0 / (b_axis * b_axis)

    a = (dx**2 + dy**2) * inv_a2 + (dz**2) * inv_b2
    b = 2.0 * ((ox * dx + oy * dy) * inv_a2 + (oz * dz) * inv_b2)
    c = (ox**2 + oy**2) * inv_a2 + (oz**2) * inv_b2 - 1.0

    disc = b**2 - 4.0 * a * c
    if disc < 0:
        return None
    
    sqrt_disc = np.sqrt(disc)
    u1, u2 = (-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)
    u = u1 if u1 > 0 else u2
    
    return origin + u * direction if u > 0 else None


def _apply_atmospheric_refraction(ray_world: np.ndarray, sat_pos: np.ndarray, 
                                  ground_height: float, geo: GeodeticContext) -> np.ndarray:
    """Apply a small Noerdlinger-style refraction bend towards nadir."""
    ray = np.asarray(ray_world, dtype=np.float64)
    ray_norm = np.linalg.norm(ray)
    if ray_norm <= 0.0:
        return ray_world
    ray /= ray_norm

    sat_norm = np.linalg.norm(sat_pos)
    if sat_norm <= 0.0:
        return ray_world

    nadir = -sat_pos / sat_norm
    cos_off = float(np.clip(np.dot(ray, nadir), -1.0, 1.0))
    off_nadir = float(np.arccos(cos_off))
    
    if off_nadir < 1e-8:
        return ray_world

    height_factor = float(np.exp(-max(0.0, ground_height) / geo.atmos_scale_height))
    bend = geo.atmos_beta * np.tan(off_nadir) * height_factor
    bend = float(np.clip(bend, 0.0, off_nadir * 0.9))
    
    if bend <= 0.0:
        return ray_world

    transverse = ray - cos_off * nadir
    t_norm = np.linalg.norm(transverse)
    if t_norm <= 1e-12:
        return ray_world

    t_hat = transverse / t_norm
    off_corr = off_nadir - bend
    
    ray_corr = np.cos(off_corr) * nadir + np.sin(off_corr) * t_hat
    return ray_corr / np.linalg.norm(ray_corr)


def create_model(
    aocs_path: Path,
    metadata_path: Optional[Path] = None,
    *,
    f: float = 105454.0,
    cx: float = 2048.0,
    cy: float = 2048.0,
    model_class: type = PhiSatPushbroomModel,
) -> PhiSatPushbroomModel:
    """Instantiate a sensor model from plain paths, load AOCS + timing."""

    try:
        from pipeline.utils import load_metadata_timing
    except ImportError:
        logger.warning("Failed to import load_metadata_timing. Timing will rely purely on AOCS.")
        load_metadata_timing = None

    model = model_class(f=f, cx=cx, cy=cy)

    timing = None
    if metadata_path and metadata_path.exists() and load_metadata_timing:
        timing = load_metadata_timing(str(metadata_path))
        if timing:
            model.line_time = timing["line_time"]
            logger.info("Updated line_time to %.6f s", model.line_time)

        try:
            with open(metadata_path) as f_meta:
                meta = json.load(f_meta)
            session_key = next(iter(meta))
            scene_height = meta[session_key]["Scene 0"]["SceneStart"]["SceneHeight"]
            if hasattr(model, "set_row_range") and scene_height is not None:
                h = int(scene_height)
                if h > 1:
                    model.set_row_range(0.0, float(h - 1))
        except (KeyError, IndexError, TypeError, ValueError, StopIteration):
            pass

    model.load_aocs_metadata(str(aocs_path))

    if timing and model.t0 is not None:
        shift = (timing["image_start_utc"] - model.t0 - (0 - model.cy) * model.line_time)
        model.along_shift = shift
        logger.info("Calculated along_shift: %.3f s", shift)

    return model