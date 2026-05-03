from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from pipeline.sensor_model import RobustModel, create_model
from pipeline.utils import load_calibration


def _rpc_terms(lon_n: np.ndarray, lat_n: np.ndarray, h_n: np.ndarray) -> np.ndarray:
    """Standard RPC00 20-term monomial vector."""
    L = lon_n
    P = lat_n
    H = h_n
    return np.column_stack([
        np.ones_like(L),
        L,
        P,
        H,
        L * P,
        L * H,
        P * H,
        L * L,
        P * P,
        H * H,
        L * P * H,
        L * L * L,
        L * P * P,
        L * H * H,
        L * L * P,
        P * P * P,
        P * H * H,
        L * L * H,
        P * P * H,
        H * H * H,
    ]).astype(np.float64)



def _fit_rational_rpc_component(target_n: np.ndarray, terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit normalized RPC component with den[0]=1 via linear least squares."""
    A = np.hstack([terms, -target_n[:, None] * terms[:, 1:]])
    coeff, *_ = np.linalg.lstsq(A, target_n, rcond=None)


    num = coeff[:20]
    den = np.zeros(20, dtype=np.float64)
    den[0] = 1.0
    den[1:] = coeff[20:]
    return num, den

def _get_dem_height_range(dem_path: Path) -> tuple[float, float]:
    """Return (h_min, h_max) from DEM valid pixels for the whole scene."""
    fallback = (-500.0, 5000.0)

    if dem_path is None or not dem_path.exists():
        return fallback

    try:
        with rasterio.open(str(dem_path)) as dem_src:
            dem = dem_src.read(1).astype(np.float64)
            if dem_src.nodata is not None:
                dem[dem == float(dem_src.nodata)] = np.nan

        finite_dem = dem[np.isfinite(dem)]
        if finite_dem.size < 100:
            return fallback

        h_lo = float(np.min(finite_dem))
        h_hi = float(np.max(finite_dem))
        if not (np.isfinite(h_lo) and np.isfinite(h_hi)):
            return fallback

        if h_hi <= h_lo:
            return fallback

        return h_lo, h_hi
    except Exception:
        return fallback



def fit_rpc(
    phisat_tiff: Path,
    calibration_path: Path,
    dem_path: Path,
    output_path: Path,
    aocs_path: Path, 
    metadata_path: Optional[Path] = None,
    grid_size: int = 80,
    f: float = 105454.0,
    cx: float = 2048.0,
    cy: float = 2048.0,
) -> Path:
    """Fit RPC coefficients from calibrated rigorous model and DEM height span."""
    if not phisat_tiff.exists():
        raise FileNotFoundError(f"PhiSat image not found: {phisat_tiff}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calibration_path}")
    if not aocs_path.exists():
        raise FileNotFoundError(f"AOCS not found: {aocs_path}")

    model = create_model(
        aocs_path,
        metadata_path,
        f=f,
        cx=cx,
        cy=cy,
        model_class=RobustModel,
    )
    calib = load_calibration(str(calibration_path))

    params = {
        "time_shift": float(calib.get("time_shift", 0.0)),
        "boresight_roll": float(calib.get("boresight_roll", calib.get("roll", 0.0))),
        "boresight_pitch": float(calib.get("boresight_pitch", calib.get("pitch", 0.0))),
        "boresight_yaw": float(calib.get("boresight_yaw", calib.get("yaw", 0.0))),
        "f_scale": float(calib.get("f_scale", float(calib.get("f", model.f)) / float(model.f))),
        "k1": float(calib.get("k1", 0.0)),
        "k2": float(calib.get("k2", 0.0)),
        "cx_bias": float(calib.get("cx_bias", 0.0)),
        "cy_bias": float(calib.get("cy_bias", 0.0)),
        "drift_roll_1": float(calib.get("drift_roll_1", calib.get("roll_rate", 0.0))),
        "drift_pitch_1": float(calib.get("drift_pitch_1", calib.get("pitch_rate", 0.0))),
        "drift_yaw_1": float(calib.get("drift_yaw_1", calib.get("yaw_rate", 0.0))),
        "cx_rate": float(calib.get("cx_rate", 0.0)),
        "along_rate": float(calib.get("along_rate", 0.0)),
    }


    with rasterio.open(str(phisat_tiff)) as src:
        image_w = int(src.width)
        image_h = int(src.height)

    # Uniform image-plane sampling in float64 for numerically stable fitting.
    xs = np.linspace(0.0, image_w - 1.0, grid_size, dtype=np.float64)
    ys = np.linspace(0.0, image_h - 1.0, grid_size, dtype=np.float64)

    # 3D grid in DEM-derived global height range for this scene.
    h_min, h_max = _get_dem_height_range(dem_path)
    height_levels = np.linspace(h_min, h_max, 5, dtype=np.float64)
    print(f"Using DEM height range: [{h_min:.3f}, {h_max:.3f}] m")
    print(f"Height levels: {height_levels}")

    u_list: list[float] = []
    v_list: list[float] = []
    lon_list: list[float] = []
    lat_list: list[float] = []
    h_list: list[float] = []

    for h_i in height_levels:
        for v in ys:
            for u in xs:
                ecef = model.predict_with_params(
                    float(u), float(v), params, ground_height=float(h_i)
                )
                if ecef is None:
                    continue

                lon_i, lat_i, _ = model.ecef_to_lonlat(*ecef)
                if not (np.isfinite(lon_i) and np.isfinite(lat_i)):
                    continue

                u_list.append(float(u))
                v_list.append(float(v))
                lon_list.append(float(lon_i))
                lat_list.append(float(lat_i))
                h_list.append(float(h_i))


    u = np.asarray(u_list, dtype=np.float64)
    v = np.asarray(v_list, dtype=np.float64)
    lon = np.asarray(lon_list, dtype=np.float64)
    lat = np.asarray(lat_list, dtype=np.float64)
    h = np.asarray(h_list, dtype=np.float64)


    if len(u) < max(100, grid_size * grid_size // 8):
        raise RuntimeError(f"Not enough valid samples for RPC fitting: {len(u)}")


    # 5) Normalize to [-1, 1]
    line_off = 0.5 * (float(v.min()) + float(v.max()))
    samp_off = 0.5 * (float(u.min()) + float(u.max()))
    line_scale = max(0.5 * (float(v.max()) - float(v.min())), 1e-8)
    samp_scale = max(0.5 * (float(u.max()) - float(u.min())), 1e-8)


    lat_off = 0.5 * (float(lat.min()) + float(lat.max()))
    lon_off = 0.5 * (float(lon.min()) + float(lon.max()))
    lat_scale = max(0.5 * (float(lat.max()) - float(lat.min())), 1e-8)
    lon_scale = max(0.5 * (float(lon.max()) - float(lon.min())), 1e-8)


    height_off = 0.5 * (float(h.min()) + float(h.max()))
    height_scale = max(0.5 * (float(h.max()) - float(h.min())), 1e-8)


    lon_n = (lon - lon_off) / lon_scale
    lat_n = (lat - lat_off) / lat_scale
    h_n = (h - height_off) / height_scale


    # 6) Standard 20-term RPC basis
    terms = _rpc_terms(lon_n, lat_n, h_n)


    line_n = (v - line_off) / line_scale
    samp_n = (u - samp_off) / samp_scale


    # 7-8) Least-squares estimation of full rational RPC coefficients
    line_num, line_den = _fit_rational_rpc_component(line_n, terms)
    samp_num, samp_den = _fit_rational_rpc_component(samp_n, terms)


    rpc = {
        "LINE_OFF": line_off,
        "SAMP_OFF": samp_off,
        "LAT_OFF": lat_off,
        "LONG_OFF": lon_off,
        "HEIGHT_OFF": height_off,
        "LINE_SCALE": line_scale,
        "SAMP_SCALE": samp_scale,
        "LAT_SCALE": lat_scale,
        "LONG_SCALE": lon_scale,
        "HEIGHT_SCALE": height_scale,
        "LINE_NUM_COEFF": [float(x) for x in line_num],
        "LINE_DEN_COEFF": [float(x) for x in line_den],
        "SAMP_NUM_COEFF": [float(x) for x in samp_num],
        "SAMP_DEN_COEFF": [float(x) for x in samp_den],
    }


    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rpc, indent=2))


    den_line = terms @ line_den
    den_samp = terms @ samp_den
    pred_line_n = (terms @ line_num) / den_line
    pred_samp_n = (terms @ samp_num) / den_samp
    pred_v = pred_line_n * line_scale + line_off
    pred_u = pred_samp_n * samp_scale + samp_off
    rmse_pix = float(np.sqrt(np.mean((pred_u - u) ** 2 + (pred_v - v) ** 2)))


    print(f"Saved RPC JSON: {out_path}")
    print(f"Samples used: {len(u)}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Fit RMSE (px): {rmse_pix:.4f}")
    return out_path