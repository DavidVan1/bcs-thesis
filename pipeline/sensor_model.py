"""
PhiSat-2 Pushbroom Sensor Model
================================

Forward model:  pixel (px, py) → ground coordinates (lon, lat)
Inverse model:  (lon, lat) → pixel (px, py)  (via Newton-Raphson)

Supports:
- Pushbroom geometry (time-dependent position per scanline)
- Ray–ellipsoid intersection (WGS-84)
- LVLH orbital frame
- Mounting bias (roll, pitch, yaw)
- Radial distortion (k1, k2)
- DEM-aware forward projection for calibration
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from scipy.spatial.transform import Rotation as SciRot


# ═══════════════════════════════════════════════════════════════════════════
# WGS-84 ellipsoid constants
# ═══════════════════════════════════════════════════════════════════════════

WGS84_A = 6_378_137.0          # semi-major axis  (m)
WGS84_B = 6_356_752.314245     # semi-minor axis  (m)
WGS84_E2 = 1.0 - (WGS84_B / WGS84_A) ** 2   # first eccentricity squared

R_EARTH = 6_371_000.0  # spherical fallback (kept for backward compat)


def _geodetic_to_ecef(lon_deg: float, lat_deg: float,
                      h: float = 0.0) -> np.ndarray:
    """Convert geodetic (lon, lat, height) to ECEF using WGS-84."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + h) * cos_lat * np.cos(lon)
    y = (N + h) * cos_lat * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z])


def _ecef_to_geodetic(x: float, y: float,
                      z: float) -> Tuple[float, float, float]:
    """ECEF → geodetic (lon_deg, lat_deg, height) via Bowring iteration."""
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x * x + y * y)
    # Initial estimate
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        lat = np.arctan2(z + WGS84_E2 * N * sin_lat, p)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    if abs(cos_lat) > 1e-10:
        h = p / cos_lat - N
    else:
        h = abs(z) / abs(sin_lat) - N * (1.0 - WGS84_E2)
    return lon, np.degrees(lat), h


# ═══════════════════════════════════════════════════════════════════════════
# Ray–ellipsoid intersection
# ═══════════════════════════════════════════════════════════════════════════

def _intersect_ellipsoid(origin: np.ndarray, direction: np.ndarray,
                         h: float = 0.0) -> Optional[np.ndarray]:
    """
    Ray–ellipsoid intersection for WGS-84 inflated by height *h*.

    The ellipsoid is  (x/a)² + (y/a)² + (z/b)² = 1  with
    a = WGS84_A + h,  b = WGS84_B + h.
    """
    a = WGS84_A + h
    b = WGS84_B + h
    inv = np.array([1.0 / a, 1.0 / a, 1.0 / b])
    o = origin * inv
    d = direction * inv
    A = np.dot(d, d)
    B = 2.0 * np.dot(o, d)
    C = np.dot(o, o) - 1.0
    disc = B * B - 4.0 * A * C
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)
    t = t1 if t1 > 0 else t2
    if t <= 0:
        return None
    return origin + t * direction


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

        return _intersect_ellipsoid(sat_pos, ray_world, h=ground_height)

    # ── ECEF ↔ geodetic ────────────────────────────────────────────

    @staticmethod
    def ecef_to_lonlat(x: float, y: float,
                       z: float) -> Tuple[float, float, float]:
        """ECEF → (lon, lat, height) using WGS-84 Bowring iteration."""
        return _ecef_to_geodetic(x, y, z)

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

    Parameter vector (10):
        [time_shift, roll, pitch, yaw, f_scale, k1, k2,
         dcx, along_scale, dcy]

    dcx / dcy are offsets from the nominal principal point (pixels).
    along_scale multiplies the base line_time for along-track correction.

    The model optionally queries a DEM callable for terrain height,
    then iterates ray–ellipsoid intersection to converge on the
    terrain surface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dem_fn = None   # callable(lon, lat) → height or None

    def set_dem(self, dem_fn) -> None:
        """Attach a DEM query function: dem_fn(lon, lat) → height_m."""
        self._dem_fn = dem_fn

    def predict_with_params(self, px: float, py: float,
                            params: np.ndarray) -> Optional[np.ndarray]:
        """
        Forward-project a pixel using the current parameter vector.

        If a DEM is attached, iteratively refines ground height
        (up to 5 iterations, converging when Δh < 0.5 m).
        """
        t_shift     = params[0]
        roll        = params[1]
        pitch       = params[2]
        yaw         = params[3]
        f_scale     = params[4]
        k1          = params[5]
        k2          = params[6]
        dcx         = params[7]
        along_scale = params[8]
        dcy         = params[9]

        current_f  = self.f * f_scale
        current_cx = self.cx + dcx
        current_cy = self.cy + dcy

        # --- Normalised image coordinate (cross-track only for pushbroom)
        x_norm = (px - current_cx) / current_f
        y_norm = 0.0

        r2 = x_norm ** 2 + y_norm ** 2
        dist = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x_norm * dist
        yd = y_norm * dist

        ray_cam = np.array([yd, xd, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Mounting bias
        rot = SciRot.from_euler("xyz", [roll, pitch, yaw], degrees=True)
        ray_body = rot.apply(ray_cam)

        # Satellite position at scanline time (along_scale corrects line_time)
        effective_row0 = current_cy
        dt = ((py - effective_row0) * self.line_time * along_scale
              + self.along_shift)
        t = self.t0 + dt + t_shift
        sat_pos = self.position + self.velocity * (t - self.t0)

        # Orbital frame
        R_b2o = self.quaternion_to_rotation_matrix(self.quaternion)
        R_o2e = self.get_orbital_rotation_matrix(sat_pos, self.velocity)
        ray_world = R_o2e @ R_b2o @ ray_body

        # --- Iterative DEM-aware intersection ---
        if self._dem_fn is not None:
            h = 0.0
            for _ in range(5):
                pt = _intersect_ellipsoid(sat_pos, ray_world, h=h)
                if pt is None:
                    return None
                lon, lat, _ = _ecef_to_geodetic(*pt)
                h_new = self._dem_fn(lon, lat)
                if h_new is None:
                    break          # outside DEM → keep last h
                if abs(h_new - h) < 0.5:
                    h = h_new
                    break
                h = h_new
            return _intersect_ellipsoid(sat_pos, ray_world, h=h)
        else:
            # Fallback: ellipsoid at sea level
            return _intersect_ellipsoid(sat_pos, ray_world, h=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers (backward compat)
# ═══════════════════════════════════════════════════════════════════════════

def _intersect_sphere(origin: np.ndarray, direction: np.ndarray,
                      radius: float) -> Optional[np.ndarray]:
    """Ray–sphere intersection (legacy fallback).  Returns ECEF or None."""
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
