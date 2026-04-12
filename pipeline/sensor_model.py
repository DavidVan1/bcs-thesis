"""
PhiSat-2 Pushbroom Sensor Model
================================

Forward model:  pixel (px, py) → ground coordinates (lon, lat)
Inverse model:  (lon, lat) → pixel (px, py)  (via Newton-Raphson)

Supports:
- Pushbroom geometry (time-dependent position per scanline)
- Ray–ellipsoid intersection with WGS84 Earth
- LVLH orbital frame
- Mounting bias (roll, pitch, yaw)
- Radial distortion (k1, k2)
- Optional atmospheric refraction correction (Noerdlinger-style)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from pyproj import Transformer
from scipy.spatial.transform import Rotation as SciRot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Base pushbroom model
# ═══════════════════════════════════════════════════════════════════════════

WGS84_A = 6_378_137.0                # semi-major axis [m]
WGS84_B = 6_356_752.314245           # semi-minor axis [m]
ATMOS_REFRACTION_BETA = 2.0e-5       # rad-scale (Noerdlinger-style approximation)
ATMOS_SCALE_HEIGHT_M = 8_000.0       # exponential density scale height [m]


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

    _GEODETIC_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    _ECEF_TO_GEODETIC = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

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

        # logger.info("Loaded AOCS metadata:")
        # logger.info("  Position: %s", self.position)
        # logger.info("  Velocity: %s", self.velocity)

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

        # Body-frame convention used by orthorectify/calibration:
        # X = along-track, Y = cross-track, Z = boresight
        # Pushbroom has no instantaneous along-track angle -> X=0.
        ray_cam = np.array([0.0, xd, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        t = self.scanline_to_time(py)
        sat_pos = self.position + self.velocity * (t - self.t0)

        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)

        ray_world = R_o2e @ R_b2o @ ray_cam
        ray_world = _apply_atmospheric_refraction(
            ray_world=ray_world,
            sat_pos=sat_pos,
            ground_height=float(ground_height),
        )

        return _intersect_ellipsoid(
            origin=sat_pos,
            direction=ray_world,
            ground_height=float(ground_height),
        )

    # ── ECEF ↔ geodetic ────────────────────────────────────────────

    @staticmethod
    def ecef_to_lonlat(x: float, y: float,
                       z: float) -> Tuple[float, float, float]:
        """ECEF → (lon, lat, height) using WGS84 ellipsoid."""
        lon, lat, h = PhiSatPushbroomModel._ECEF_TO_GEODETIC.transform(
            float(x), float(y), float(z)
        )
        return float(lon), float(lat), float(h)

    @staticmethod
    def lonlat_to_ecef(lon: float, lat: float,
                       height: float = 0.0) -> np.ndarray:
        """(lon, lat, height) → ECEF using WGS84 ellipsoid."""
        x, y, z = PhiSatPushbroomModel._GEODETIC_TO_ECEF.transform(
            float(lon), float(lat), float(height)
        )
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

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

    Parameter vector (9 or 12):
        [time_shift, roll, pitch, yaw, f_scale, k1, k2, cx_rate, along_rate]
        [roll_rate, pitch_rate, yaw_rate]   ← optional linear drift terms

    cx_rate models a linear cross-track drift: the effective principal
    point shifts as  cx_eff = cx + cx_rate · (py − cy).

    along_rate corrects a linear along-track ephemeris error by scaling
    the orbit propagation time:  sat_pos = pos + vel · dt · (1 + along_rate).

    roll_rate / pitch_rate / yaw_rate model linear attitude drift during
    acquisition.  The scanline position is normalised to [−0.5, 0.5] so
    the rate parameters have the same unit (degrees) as the constant bias
    terms and bounds can be applied symmetrically:

        effective_roll(v)  = roll  + roll_rate  · norm_v
        effective_pitch(v) = pitch + pitch_rate · norm_v
        effective_yaw(v)   = yaw   + yaw_rate   · norm_v

    where  norm_v = (v − cy) / image_height.

    This single extra degree of freedom per axis is enough to absorb the
    ~0.004° attitude drift that causes the observed north-to-south RMSE
    gradient without over-fitting.
    """

    def predict_with_params(self, px: float, py: float,
                            params: np.ndarray,
                            ground_height: float = 0.0) -> Optional[np.ndarray]:
        """Forward-project a pixel using the current parameter vector."""
        t_shift, r, p, y = params[0], params[1], params[2], params[3]
        f_scale = params[4]
        k1, k2 = params[5], params[6]
        cx_rate    = params[7] if len(params) > 7  else 0.0
        along_rate = params[8] if len(params) > 8  else 0.0
        roll_rate  = params[9] if len(params) > 9  else 0.0
        pitch_rate = params[10] if len(params) > 10 else 0.0
        yaw_rate   = params[11] if len(params) > 11 else 0.0

        current_f = self.f * f_scale

        # Normalised scanline position in [−0.5, 0.5]
        # Uses image height derived from cy (principal point ≈ image centre)
        norm_v = (py - self.cy) / (2.0 * self.cy) if self.cy > 0 else 0.0

        # Linear attitude drift: effective angles vary with scanline
        r_eff = r + roll_rate  * norm_v
        p_eff = p + pitch_rate * norm_v
        y_eff = y + yaw_rate   * norm_v

        # Cross-track drift: effective cx shifts linearly with scanline
        cx_eff = self.cx + cx_rate * (py - self.cy)
        x_norm = (px - cx_eff) / current_f
        y_norm = 0.0

        r2 = x_norm**2 + y_norm**2
        dist = 1.0 + k1 * r2 + k2 * r2 * r2
        xd, yd = x_norm * dist, y_norm * dist

        # Keep identical axis convention as OrthorectificationEngine.
        ray_cam = np.array([0.0, xd, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Mounting bias + linear drift
        rot = SciRot.from_euler("xyz", [r_eff, p_eff, y_eff], degrees=True)
        ray_body = rot.apply(ray_cam)

        t = self.scanline_to_time(py) + t_shift
        dt = t - self.t0
        sat_pos = self.position + self.velocity * dt * (1.0 + along_rate)

        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)

        ray_world = R_o2e @ R_b2o @ ray_body
        ray_world = _apply_atmospheric_refraction(
            ray_world=ray_world,
            sat_pos=sat_pos,
            ground_height=float(ground_height),
        )
        return _intersect_ellipsoid(
            origin=sat_pos,
            direction=ray_world,
            ground_height=float(ground_height),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _intersect_ellipsoid(origin: np.ndarray,
                         direction: np.ndarray,
                         ground_height: float = 0.0) -> Optional[np.ndarray]:
    """Ray–WGS84-ellipsoid intersection at constant ellipsoidal height."""
    a_axis = WGS84_A + float(ground_height)
    b_axis = WGS84_B + float(ground_height)

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

    inv_a2 = 1.0 / (a_axis * a_axis)
    inv_b2 = 1.0 / (b_axis * b_axis)

    a = (dx * dx + dy * dy) * inv_a2 + (dz * dz) * inv_b2
    b = 2.0 * ((ox * dx + oy * dy) * inv_a2 + (oz * dz) * inv_b2)
    c = (ox * ox + oy * oy) * inv_a2 + (oz * oz) * inv_b2 - 1.0

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


def _apply_atmospheric_refraction(ray_world: np.ndarray,
                                  sat_pos: np.ndarray,
                                  ground_height: float = 0.0) -> np.ndarray:
    """Apply a small Noerdlinger-style refraction bend towards nadir.

    This keeps a lightweight physically-motivated correction without a full
    ray-tracing atmosphere model. The bend magnitude scales with off-nadir
    angle and exponentially with target height.
    """
    ray = np.asarray(ray_world, dtype=np.float64)
    ray_norm = np.linalg.norm(ray)
    if ray_norm <= 0.0:
        return ray_world
    ray /= ray_norm

    sat = np.asarray(sat_pos, dtype=np.float64)
    sat_norm = np.linalg.norm(sat)
    if sat_norm <= 0.0:
        return ray_world

    nadir = -sat / sat_norm
    cos_off = float(np.clip(np.dot(ray, nadir), -1.0, 1.0))
    off_nadir = float(np.arccos(cos_off))
    if off_nadir < 1e-8:
        return ray_world

    # Noerdlinger-style lightweight approximation: bend towards nadir.
    # Stronger at larger off-nadir, weaker at higher terrain elevation.
    height_factor = float(np.exp(-max(0.0, ground_height) / ATMOS_SCALE_HEIGHT_M))
    bend = ATMOS_REFRACTION_BETA * np.tan(off_nadir) * height_factor
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
    ray_corr /= np.linalg.norm(ray_corr)
    return ray_corr


# ═══════════════════════════════════════════════════════════════════════════
# Model factory (convenience for other modules)
# ═══════════════════════════════════════════════════════════════════════════

def create_model(config: "SceneConfig",
                 model_class: type = PhiSatPushbroomModel,
                 ) -> PhiSatPushbroomModel:
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

    # Load timing from metadata once
    timing = None
    if config.metadata_path and config.metadata_path.exists():
        timing = load_metadata_timing(str(config.metadata_path))
        if timing:
            model.line_time = timing["line_time"]
            logger.info("Updated line_time to %.6f s", model.line_time)

    # AOCS
    model.load_aocs_metadata(str(config.aocs_path))

    # Along-shift from metadata (reuse already-loaded timing)
    if timing and model.t0 is not None:
        shift = (timing["image_start_utc"]
                 - model.t0
                 - (0 - model.cy) * model.line_time)
        model.along_shift = shift
        logger.info("Calculated along_shift: %.3f s", shift)

    return model