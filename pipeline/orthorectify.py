"""
DEM-aware orthorectification of PhiSat-2 pushbroom imagery.

Pipeline:
  1. Forward-project image corners to compute ground footprint.
  2. Build a sparse lookup table (LUT) via Newton back-projection.
  3. Interpolate LUT to full resolution.
  4. Resample raw image with cv2.remap.
  5. Write orthorectified GeoTIFF.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple

import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as SciRot

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from scipy.ndimage import map_coordinates

from .config import SceneConfig
from .sensor_model import PhiSatPushbroomModel, create_model
from .utils import load_calibration


# ═══════════════════════════════════════════════════════════════════════════

class OrthorectificationEngine:
    """
    Orthorectify pushbroom imagery using a calibrated sensor model and DEM.
    """

    def __init__(self, model: PhiSatPushbroomModel,
                 dem_path: str,
                 phisat_image_path: str,
                 calib_params: dict):
        self.model = model
        self.calib_params = calib_params

        # DEM
        self.dem_src = rasterio.open(dem_path)
        self.dem_data = self.dem_src.read(1)
        self.dem_crs = self.dem_src.crs
        self.dem_transform = self.dem_src.transform
        self.dem_bounds = self.dem_src.bounds

        # Raw image
        self.phisat_src = rasterio.open(phisat_image_path)
        self.phisat_data = self.phisat_src.read()
        self.phisat_shape = self.phisat_data.shape

        print(f"DEM shape      : {self.dem_data.shape}")
        print(f"DEM CRS        : {self.dem_crs}")
        print(f"PhiSat shape   : {self.phisat_shape}")

        self._apply_calibration()

    # ── calibration ─────────────────────────────────────────────────

    def _apply_calibration(self):
        p = self.calib_params
        self.model.f = p.get("f", self.model.f)
        self.model.cx = p.get("cx", self.model.cx)
        self.model.cy = p.get("cy", self.model.cy)
        self.model.k1 = p.get("k1", 0.0)
        self.model.k2 = p.get("k2", 0.0)
        self.calib_time_shift = p.get("time_shift", 0.0)
        self.calib_along_scale = p.get("along_scale", 1.0)

        self.mounting_bias = {
            "roll": p.get("roll", 0.0),
            "pitch": p.get("pitch", 0.0),
            "yaw": p.get("yaw", 0.0),
        }
        self._rot_bias = SciRot.from_euler("xyz", [
            self.mounting_bias["roll"],
            self.mounting_bias["pitch"],
            self.mounting_bias["yaw"],
        ], degrees=True)

        self._R_b2o = self.model.quaternion_to_rotation_matrix(
            self.model.quaternion)

        print(f"Calibration: f={self.model.f:.1f}  "
              f"cx={self.model.cx:.2f}  cy={self.model.cy:.2f}  "
              f"R={self.mounting_bias['roll']:.3f}°  "
              f"P={self.mounting_bias['pitch']:.3f}°  "
              f"Y={self.mounting_bias['yaw']:.3f}°  "
              f"along_scale={self.calib_along_scale:.6f}")

    # ── DEM query ───────────────────────────────────────────────────

    def dem_query(self, lon: float, lat: float) -> Optional[float]:
        """Bilinear DEM elevation at (lon, lat)."""
        px = (lon - self.dem_transform.c) / self.dem_transform.a
        py = (lat - self.dem_transform.f) / self.dem_transform.e
        if (px < 0 or px >= self.dem_data.shape[1]
                or py < 0 or py >= self.dem_data.shape[0]):
            return None
        ix, iy = int(np.floor(px)), int(np.floor(py))
        if ix + 1 >= self.dem_data.shape[1] or iy + 1 >= self.dem_data.shape[0]:
            return None
        fx, fy = px - ix, py - iy
        z = ((1 - fx) * (1 - fy) * self.dem_data[iy, ix]
             + fx * (1 - fy) * self.dem_data[iy, ix + 1]
             + (1 - fx) * fy * self.dem_data[iy + 1, ix]
             + fx * fy * self.dem_data[iy + 1, ix + 1])
        return float(z)

    # ── forward projection with bias ────────────────────────────────

    def _forward_biased(self, u: float, v: float) -> Optional[np.ndarray]:
        """Forward-project pixel (u, v) → ECEF using calibrated model.

        Uses iterative DEM-aware ray–ellipsoid intersection so that
        the Newton back-projection converges to the correct ground point
        at terrain height (not sea level).
        """
        from .sensor_model import _intersect_ellipsoid, _ecef_to_geodetic
        x_norm = (u - self.model.cx) / self.model.f
        r2 = x_norm ** 2
        dist = 1.0 + self.model.k1 * r2 + self.model.k2 * r2 ** 2
        ray_cam = np.array([0.0, x_norm * dist, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        ray_body = self._rot_bias.apply(ray_cam)

        # Along-track timing with calibrated scale
        dt = ((v - self.model.cy) * self.model.line_time
              * self.calib_along_scale + self.model.along_shift)
        t = self.model.t0 + dt + self.calib_time_shift
        sat_pos = self.model.position + self.model.velocity * (t - self.model.t0)

        R_o2e = self.model.get_orbital_rotation_matrix(
            sat_pos, self.model.velocity)
        ray_world = R_o2e @ self._R_b2o @ ray_body

        # Iterative DEM-aware intersection (same as _forward_lonlat)
        h = 0.0
        for _ in range(5):
            pt = _intersect_ellipsoid(sat_pos, ray_world, h=h)
            if pt is None:
                return None
            lon, lat, _ = _ecef_to_geodetic(*pt)
            h_new = self.dem_query(lon, lat)
            if h_new is None:
                break
            if abs(h_new - h) < 0.5:
                h = h_new
                break
            h = h_new
        return _intersect_ellipsoid(sat_pos, ray_world, h=h)

    def _forward_lonlat(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Forward pixel → (lon, lat) with iterative DEM refinement."""
        from .sensor_model import _intersect_ellipsoid, _ecef_to_geodetic
        x_norm = (u - self.model.cx) / self.model.f
        r2 = x_norm ** 2
        dist = 1.0 + self.model.k1 * r2 + self.model.k2 * r2 ** 2
        ray_cam = np.array([0.0, x_norm * dist, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_body = self._rot_bias.apply(ray_cam)

        dt = ((v - self.model.cy) * self.model.line_time
              * self.calib_along_scale + self.model.along_shift)
        t = self.model.t0 + dt + self.calib_time_shift
        sat_pos = self.model.position + self.model.velocity * (t - self.model.t0)

        R_o2e = self.model.get_orbital_rotation_matrix(
            sat_pos, self.model.velocity)
        ray_world = R_o2e @ self._R_b2o @ ray_body

        # Iterative DEM-aware intersection
        h = 0.0
        for _ in range(5):
            pt = _intersect_ellipsoid(sat_pos, ray_world, h=h)
            if pt is None:
                return None
            lon, lat, _ = _ecef_to_geodetic(*pt)
            h_new = self.dem_query(lon, lat)
            if h_new is None:
                break
            if abs(h_new - h) < 0.5:
                h = h_new
                break
            h = h_new
        pt = _intersect_ellipsoid(sat_pos, ray_world, h=h)
        if pt is None:
            return None
        lon, lat, _ = _ecef_to_geodetic(*pt)
        return lon, lat

    # ── back-projection (Newton) ────────────────────────────────────

    def ground_to_image(self, lon: float, lat: float,
                        v_hint: Optional[float] = None
                        ) -> Optional[Tuple[float, float]]:
        """Back-project (lon, lat) → image pixel (u, v)."""
        from .sensor_model import _geodetic_to_ecef
        height = self.dem_query(lon, lat)
        if height is None:
            return None

        ground = _geodetic_to_ecef(lon, lat, height)

        return self._newton(ground, v_hint)

    def _newton(self, ground_ecef: np.ndarray,
                v_hint: Optional[float] = None,
                max_iter: int = 20, tol: float = 1.0
                ) -> Optional[Tuple[float, float]]:
        """Newton–Raphson back-projection."""

        def _solve(u0, v0):
            u, v = float(u0), float(v0)
            for _ in range(max_iter):
                proj = self._forward_biased(u, v)
                if proj is None:
                    return None
                res = proj - ground_ecef
                if np.linalg.norm(res) < tol:
                    if (0 <= u < self.phisat_shape[2]
                            and 0 <= v < self.phisat_shape[1]):
                        return (u, v)
                    return None
                d = 0.25
                pp = self._forward_biased(u + d, v)
                pm = self._forward_biased(u - d, v)
                vp = self._forward_biased(u, v + d)
                vm = self._forward_biased(u, v - d)
                if any(x is None for x in (pp, pm, vp, vm)):
                    return None
                du = (pp - pm) / (2 * d)
                dv = (vp - vm) / (2 * d)
                J = np.column_stack([du, dv])
                try:
                    delta = -np.linalg.lstsq(J, res, rcond=None)[0]
                    u += delta[0]
                    v += delta[1]
                except Exception:
                    return None
            if (0 <= u < self.phisat_shape[2]
                    and 0 <= v < self.phisat_shape[1]):
                return (u, v)
            return None

        # Quick search around hint
        if v_hint is not None:
            vr = max(10, (self.phisat_shape[1] - 1) / 100)
            best_v, best_d = v_hint, float("inf")
            for vt in np.linspace(max(0, v_hint - vr),
                                  min(self.phisat_shape[1] - 1, v_hint + vr), 5):
                p = self._forward_biased(self.model.cx, vt)
                if p is not None:
                    d = np.linalg.norm(p - ground_ecef)
                    if d < best_d:
                        best_d, best_v = d, vt
            result = _solve(self.model.cx, best_v)
            if result is not None:
                return result

        # Full 1-D line search
        n_s = 7
        vs = np.linspace(0, self.phisat_shape[1] - 1, n_s)
        best_v, best_d = self.phisat_shape[1] / 2, float("inf")
        for vt in vs:
            p = self._forward_biased(self.model.cx, vt)
            if p is not None:
                d = np.linalg.norm(p - ground_ecef)
                if d < best_d:
                    best_d, best_v = d, vt

        step = (self.phisat_shape[1] - 1) / (n_s - 1)
        for vt in np.linspace(max(0, best_v - step),
                              min(self.phisat_shape[1] - 1, best_v + step), 9):
            p = self._forward_biased(self.model.cx, vt)
            if p is not None:
                d = np.linalg.norm(p - ground_ecef)
                if d < best_d:
                    best_d, best_v = d, vt

        return _solve(self.model.cx, best_v)

    # ── main orthorectification ─────────────────────────────────────

    def orthorectify(self) -> Optional[Tuple[np.ndarray, Affine, CRS]]:
        """Run the full orthorectification.  Returns (data, transform, crs)."""
        print("\n" + "=" * 60)
        print("ORTHORECTIFICATION")
        print("=" * 60)

        # Step 1: Footprint
        print("\n[1] Computing calibrated image footprint…")
        corners = [
            (0, 0),
            (self.phisat_shape[2] - 1, 0),
            (0, self.phisat_shape[1] - 1),
            (self.phisat_shape[2] - 1, self.phisat_shape[1] - 1),
            (self.model.cx, 0),
            (self.model.cx, self.phisat_shape[1] - 1),
        ]
        lons, lats = [], []
        for px, py in corners:
            ll = self._forward_lonlat(px, py)
            if ll:
                lons.append(ll[0])
                lats.append(ll[1])
        if len(lons) < 2:
            print("  ERROR: cannot compute footprint")
            return None

        margin = 0.02
        fp_w, fp_e = min(lons) - margin, max(lons) + margin
        fp_s, fp_n = min(lats) - margin, max(lats) + margin

        # Clip to DEM
        db = self.dem_bounds
        o_w = max(fp_w, db.left)
        o_e = min(fp_e, db.right)
        o_s = max(fp_s, db.bottom)
        o_n = min(fp_n, db.top)
        if o_w >= o_e or o_s >= o_n:
            print("  ERROR: footprint outside DEM")
            return None

        res_x = abs(self.dem_transform.a)
        res_y = abs(self.dem_transform.e)
        out_w = int(np.ceil((o_e - o_w) / res_x))
        out_h = int(np.ceil((o_n - o_s) / res_y))
        print(f"  Output: {out_w}×{out_h}  res {res_x:.6f}°")

        lons_arr = np.linspace(o_w, o_e, out_w)
        lats_arr = np.linspace(o_n, o_s, out_h)

        out_tf = Affine((o_e - o_w) / out_w, 0, o_w,
                        0, -(o_n - o_s) / out_h, o_n)

        # Step 2: Sparse LUT
        print("\n[2] Building sparse LUT…")
        lut_step = 10
        rows_lut = np.arange(0, out_h, lut_step, dtype=int)
        cols_lut = np.arange(0, out_w, lut_step, dtype=int)
        if rows_lut[-1] != out_h - 1:
            rows_lut = np.append(rows_lut, out_h - 1)
        if cols_lut[-1] != out_w - 1:
            cols_lut = np.append(cols_lut, out_w - 1)

        lut_u = np.full((len(rows_lut), len(cols_lut)), np.nan, np.float32)
        lut_v = np.full_like(lut_u, np.nan)

        ok = fail = 0
        last_v_row = None
        prev_row_v = np.full(len(cols_lut), np.nan)

        for i, ri in enumerate(rows_lut):
            if i % max(1, len(rows_lut) // 10) == 0:
                print(f"    Row {i}/{len(rows_lut)} "
                      f"({100 * i / len(rows_lut):.0f}%)")
            lat = lats_arr[ri]
            last_v_row = None
            for j, ci in enumerate(cols_lut):
                lon = lons_arr[ci]
                hint = (last_v_row if last_v_row is not None
                        else (prev_row_v[j] if not np.isnan(prev_row_v[j])
                              else None))
                uv = self.ground_to_image(lon, lat, v_hint=hint)
                if uv is not None:
                    lut_u[i, j], lut_v[i, j] = uv
                    ok += 1
                    last_v_row = uv[1]
                    prev_row_v[j] = uv[1]
                else:
                    fail += 1
                    last_v_row = None

        print(f"  LUT ok={ok}  fail={fail}")

        # Step 3: Interpolate
        print("\n[3] Interpolating LUT…")

        def _fill_nan(arr):
            mask = np.isnan(arr)
            if not mask.any():
                return arr.copy()
            filled = arr.copy()
            _, idx = distance_transform_edt(mask, return_distances=True,
                                            return_indices=True)
            filled[mask] = arr[tuple(idx[:, mask])]
            return filled

        lut_u_f = _fill_nan(lut_u)
        lut_v_f = _fill_nan(lut_v)
        lut_valid = np.where(~np.isnan(lut_u), 1.0, 0.0).astype(np.float32)

        interp_u = RegularGridInterpolator(
            (rows_lut.astype(float), cols_lut.astype(float)),
            lut_u_f, method="linear", bounds_error=False, fill_value=np.nan)
        interp_v = RegularGridInterpolator(
            (rows_lut.astype(float), cols_lut.astype(float)),
            lut_v_f, method="linear", bounds_error=False, fill_value=np.nan)
        interp_valid = RegularGridInterpolator(
            (rows_lut.astype(float), cols_lut.astype(float)),
            lut_valid, method="linear", bounds_error=False, fill_value=0.0)

        rg, cg = np.meshgrid(np.arange(out_h, dtype=np.float32),
                              np.arange(out_w, dtype=np.float32),
                              indexing="ij")
        pts = np.column_stack([rg.ravel(), cg.ravel()])

        map_u = interp_u(pts).reshape(out_h, out_w).astype(np.float32)
        map_v = interp_v(pts).reshape(out_h, out_w).astype(np.float32)
        validity = interp_valid(pts).reshape(out_h, out_w)
        swath_bad = validity < 0.5
        map_u[swath_bad] = np.nan
        map_v[swath_bad] = np.nan

        # Step 4: Resample
        print("\n[4] Resampling…")
        n_bands = self.phisat_shape[0]
        ortho = np.zeros((n_bands, out_h, out_w), dtype=np.float32)

        if HAS_CV2:
            for b in range(n_bands):
                ortho[b] = cv2.remap(
                    self.phisat_data[b].astype(np.float32),
                    map_u, map_v,
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            for b in range(n_bands):
                ortho[b] = map_coordinates(
                    self.phisat_data[b].astype(np.float32),
                    [map_v, map_u], order=3, mode="constant", cval=0)

        # Mask invalid
        invalid = (np.isnan(map_u) | np.isnan(map_v)
                   | (map_u < 0) | (map_u >= self.phisat_shape[2] - 1)
                   | (map_v < 0) | (map_v >= self.phisat_shape[1] - 1))
        for b in range(n_bands):
            ortho[b][invalid] = 0

        valid_pct = 100 * np.sum(~invalid) / (out_h * out_w)
        print(f"  Valid pixels: {valid_pct:.1f}%")

        return ortho, out_tf, self.dem_crs

    # ── I/O ─────────────────────────────────────────────────────────

    @staticmethod
    def write_geotiff(data: np.ndarray, transform: Affine,
                      crs: CRS, output_path: str) -> None:
        """Write orthorectified array as GeoTIFF."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_bands, h, w = data.shape
        with rasterio.open(path, "w", driver="GTiff",
                           height=h, width=w, count=n_bands,
                           dtype=data.dtype, crs=crs,
                           transform=transform, compress="lzw") as dst:
            for b in range(n_bands):
                dst.write(data[b], b + 1)
        print(f"Saved GeoTIFF → {path}  ({w}×{h}, {n_bands} bands)")

    def close(self):
        self.dem_src.close()
        self.phisat_src.close()


# ── Convenience runner ─────────────────────────────────────────────────

def run_orthorectify(config: SceneConfig) -> Optional[str]:
    """
    Run orthorectification for a scene.  Returns output path or None.
    """
    missing = config.check_inputs("orthorectify")
    if missing:
        raise FileNotFoundError(
            "Missing files for orthorectify:\n  " + "\n  ".join(missing))

    print("=" * 60)
    print(f"ORTHORECTIFY — scene '{config.name}'")
    print("=" * 60)

    model = create_model(config)
    calib = load_calibration(str(config.calib_path))

    engine = OrthorectificationEngine(
        model,
        str(config.dem_path),
        str(config.phisat_image_path),
        calib,
    )

    result = engine.orthorectify()
    if result is None:
        engine.close()
        return None

    ortho_data, transform, crs = result
    out = str(config.ortho_path)
    engine.write_geotiff(ortho_data, transform, crs, out)
    engine.close()

    print(f"\nOrthorectification complete → {out}")
    return out
