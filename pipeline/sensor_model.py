"""
PhiSat-2 Pushbroom Sensor Model
================================

Forward model:  pixel (px, py) → ground coordinates (lon, lat)
Inverse model:  (lon, lat) → pixel (px, py)  (via Newton-Raphson)

Supports:
- Pushbroom geometry (time-dependent position per scanline)
- Ray–sphere intersection with spherical Earth
- LVLH orbital frame
- Mounting bias (roll, pitch, yaw)
- Radial distortion (k1, k2)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from scipy.spatial.transform import Rotation as SciRot


# ═══════════════════════════════════════════════════════════════════════════
# Base pushbroom model
# ═══════════════════════════════════════════════════════════════════════════

R_EARTH = 6_371_000.0  # metres (spherical approximation)


class PhiSatPushbroomModel:
    """
    Core geometric model for a pushbroom line scanner.

    Parameters
    ----------
    f : float   – focal length in pixels
    cx, cy      – principal point (image centre)
    k1, k2      – radial distortion coefficients
    """

    def __init__(self, f: float = 3000.0,
                 cx: float = 2048.0, cy: float = 2048.0,
                 k1: float = 0.0, k2: float = 0.0):
        self.f = f
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2

        # Platform state (set via load_aocs_metadata)
        self.position: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.quaternion: Optional[np.ndarray] = None
        self.t0: Optional[float] = None

        # Timing model
        self.row0 = cy
        self.line_time = 0.001        # seconds per scanline
        self.along_scale = 1.0
        self.along_shift = 0.0

    # ── Metadata loading ────────────────────────────────────────────

    def load_aocs_metadata(self, json_path: str,
                           acquisition_index: int = 0) -> None:
        """Load platform state from AOCS.json."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"AOCS file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        acq = data["Acquisitions"][acquisition_index]

        self.position = np.array([
            acq["OBCPositionX"],
            acq["OBCPositionY"],
            acq["OBCPositionZ"],
        ])
        self.velocity = np.array([
            acq["OBCVelocityX"],
            acq["OBCVelocityY"],
            acq["OBCVelocityZ"],
        ])
        self.quaternion = np.array(acq["QPointing"])   # [w, x, y, z]
        self.t0 = acq["ADCSTimeSec"] + acq["ADCSTimeNs"] * 1e-9

        print(f"Loaded AOCS metadata:")
        print(f"  Position: {self.position}")
        print(f"  Velocity: {self.velocity}")

    # ── Intrinsics ──────────────────────────────────────────────────

    def apply_radial_distortion(self, x: float,
                                y: float) -> Tuple[float, float]:
        """Apply Brown model radial distortion."""
        r2 = x * x + y * y
        factor = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return x * factor, y * factor

    # ── Reference-frame helpers ─────────────────────────────────────

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    @staticmethod
    def get_orbital_rotation_matrix(pos: np.ndarray,
                                    vel: np.ndarray) -> np.ndarray:
        """LVLH → ECEF rotation.  Z = nadir, Y = –orbit-normal, X ≈ along-track."""
        z_orb = -pos / np.linalg.norm(pos)
        y_orb = -np.cross(pos, vel)
        y_orb /= np.linalg.norm(y_orb)
        x_orb = np.cross(y_orb, z_orb)
        x_orb /= np.linalg.norm(x_orb)
        return np.column_stack([x_orb, y_orb, z_orb])

    # ── Timing ──────────────────────────────────────────────────────

    def scanline_to_time(self, row: float) -> float:
        """Scanline index → acquisition time (UTC seconds)."""
        dt = ((row - self.row0) * self.line_time * self.along_scale
              + self.along_shift)
        return self.t0 + dt

    # ── Forward projection ──────────────────────────────────────────

    def pixel_to_ground(self, px: float, py: float,
                        ground_height: float = 0.0
                        ) -> Optional[Tuple[float, float, float]]:
        """pixel → ECEF ground point (tuple) or None."""
        if self.position is None:
            raise ValueError("Must load AOCS metadata first")

        x = (px - self.cx) / self.f
        y = 0.0  # pushbroom: along-track angle is zero
        xd, yd = self.apply_radial_distortion(x, y)

        ray_cam = np.array([yd, xd, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        t = self.scanline_to_time(py)
        sat_pos = self.position + self.velocity * (t - self.t0)

        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)

        ray_world = R_o2e @ R_b2o @ ray_cam

        return _intersect_sphere(sat_pos, ray_world,
                                 R_EARTH + ground_height)

    # ── ECEF ↔ geodetic ────────────────────────────────────────────

    @staticmethod
    def ecef_to_lonlat(x: float, y: float,
                       z: float) -> Tuple[float, float, float]:
        """ECEF → (lon, lat, height) using spherical approximation."""
        r = np.sqrt(x*x + y*y + z*z)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        return lon, lat, r - R_EARTH

    # ── Convenience ─────────────────────────────────────────────────

    def predict_ground_coordinates(self, px: float,
                                   py: float
                                   ) -> Optional[Tuple[float, float]]:
        """pixel → (lon, lat) or None."""
        ecef = self.pixel_to_ground(px, py)
        if ecef is None:
            return None
        lon, lat, _ = self.ecef_to_lonlat(*ecef)
        return lon, lat


# ═══════════════════════════════════════════════════════════════════════════
# Extended model used during calibration
# ═══════════════════════════════════════════════════════════════════════════

class RobustModel(PhiSatPushbroomModel):
    """
    Pushbroom model with tuneable parameter vector used by the
    calibration optimizer.

    Parameter vector (7):
        [time_shift, roll, pitch, yaw, f_scale, k1, k2]
    """

    def predict_with_params(self, px: float, py: float,
                            params: np.ndarray) -> Optional[np.ndarray]:
        """Forward-project a pixel using the current parameter vector."""
        t_shift, r, p, y = params[0], params[1], params[2], params[3]
        f_scale = params[4]
        k1, k2 = params[5], params[6]

        current_f = self.f * f_scale

        x_norm = (px - self.cx) / current_f
        y_norm = 0.0

        r2 = x_norm**2 + y_norm**2
        dist = 1.0 + k1 * r2 + k2 * r2 * r2
        xd, yd = x_norm * dist, y_norm * dist

        ray_cam = np.array([yd, xd, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Mounting bias
        rot = SciRot.from_euler("xyz", [r, p, y], degrees=True)
        ray_body = rot.apply(ray_cam)

        t = self.scanline_to_time(py) + t_shift
        sat_pos = self.position + self.velocity * (t - self.t0)

        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)

        ray_world = R_o2e @ R_b2o @ ray_body
        return _intersect_sphere(sat_pos, ray_world, R_EARTH)


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _intersect_sphere(origin: np.ndarray, direction: np.ndarray,
                      radius: float) -> Optional[np.ndarray]:
    """Ray–sphere intersection.  Returns ECEF point or None."""
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(origin, direction)
    c = np.dot(origin, origin) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    u1 = (-b - sqrt_disc) / (2.0 * a)
    u2 = (-b + sqrt_disc) / (2.0 * a)
    u = u1 if u1 > 0 else u2
    if u <= 0:
        return None
    return origin + u * direction


# ═══════════════════════════════════════════════════════════════════════════
# Model factory (convenience for other modules)
# ═══════════════════════════════════════════════════════════════════════════

def create_model(config, model_class=PhiSatPushbroomModel) -> PhiSatPushbroomModel:
    """
    Instantiate a sensor model from a SceneConfig, load AOCS + timing.

    Parameters
    ----------
    config : SceneConfig
    model_class : PhiSatPushbroomModel or RobustModel

    Returns
    -------
    Initialised model with AOCS loaded and timing corrected.
    """
    from .utils import load_metadata_timing

    model = model_class(
        f=config.initial_f,
        cx=config.cx,
        cy=config.cy,
    )

    # Timing from metadata
    if config.metadata_path and config.metadata_path.exists():
        timing = load_metadata_timing(str(config.metadata_path))
        if timing:
            model.line_time = timing["line_time"]
            print(f"Updated line_time to {model.line_time:.6f} s")

    # AOCS
    model.load_aocs_metadata(str(config.aocs_path))

    # Along-shift from metadata
    if config.metadata_path and config.metadata_path.exists():
        timing = load_metadata_timing(str(config.metadata_path))
        if timing and model.t0 is not None:
            shift = (timing["image_start_utc"]
                     - model.t0
                     - (0 - model.cy) * model.line_time)
            model.along_shift = shift
            print(f"Calculated along_shift: {shift:.3f} s")

    return model
