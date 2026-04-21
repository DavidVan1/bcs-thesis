import logging, json, numpy as np, cv2, rasterio, os
from pathlib import Path
from typing import Tuple
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

PIXEL_SIZE = 4.75
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _get_gradient(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

def ncc_match(template: np.ndarray, image: np.ndarray) -> Tuple[float, float, float, bool]:
    res = cv2.matchTemplate(_get_gradient(image), _get_gradient(template), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    px, py = max_loc
    h, w = res.shape
    dx_s = dy_s = 0.0
    if 0 < px < w - 1:
        dx_s = 0.5 * (res[py, px+1] - res[py, px-1]) / (2 * res[py, px] - res[py, px+1] - res[py, px-1] + 1e-12)
    if 0 < py < h - 1:
        dy_s = 0.5 * (res[py+1, px] - res[py-1, px]) / (2 * res[py, px] - res[py+1, px] - res[py-1, px] + 1e-12)
    return (py + dy_s) - (h-1)/2.0, (px + dx_s) - (w-1)/2.0, max_val, (px <= 0 or px >= w-1 or py <= 0 or py >= h-1)

def extract_patch(ortho_path: Path, cb, chip_crs, res=PIXEL_SIZE, margin=200.0, off_e=0.0, off_n=0.0):
    l, b, r, t = cb.left - margin + off_e, cb.bottom - margin + off_n, cb.right + margin + off_e, cb.top + margin + off_n
    w, h = int(round((r-l)/res)), int(round((t-b)/res))
    tf = Affine(res, 0, l, 0, -res, t)
    
    with rasterio.open(str(ortho_path)) as src:
        # Proper RGB to Luminance conversion
        if src.count >= 3:
            out = []
            for b_idx in [1, 2, 3]:
                buf = np.zeros((h, w), dtype=np.float32)
                reproject(rasterio.band(src, b_idx), buf, src_transform=src.transform, src_crs=src.crs, dst_transform=tf, dst_crs=chip_crs, resampling=Resampling.bilinear)
                out.append(buf)
            patch = 0.2989 * out[0] + 0.5870 * out[1] + 0.1140 * out[2]
        else:
            patch = np.zeros((h, w), dtype=np.float32)
            reproject(rasterio.band(src, 1), patch, src_transform=src.transform, src_crs=src.crs, dst_transform=tf, dst_crs=chip_crs, resampling=Resampling.bilinear)
            
    return patch if np.count_nonzero(patch) > 0.3 * patch.size else None

def verify_ncc(ortho_path: Path, gcp_json_path: Path, gcp_chip_dir: Path, output_path: Path, min_ncc: float = 0.4, **kwargs) -> dict:
    with rasterio.open(str(ortho_path)) as src:
        ortho_bounds, ortho_crs = src.bounds, src.crs

    # Load from ALL JSON files in directory
    gcps = []
    seen_ids = set()
    for jf in gcp_json_path.parent.glob("*.json"):
        with open(jf) as f:
            for g in json.load(f)["GCP_DB"]["GCP"]:
                if g["ID"] in seen_ids: continue
                chip_p = gcp_chip_dir / f"{g['ID']}_00.TIF"
                if chip_p.exists():
                    gi = g["GCP_Info"]
                    gcps.append({"id": g["ID"], "lon": float(gi["Longitude"]), "lat": float(gi["Latitude"]), "chip_path": chip_p})
                    seen_ids.add(g["ID"])

    results = []
    for g in gcps:
        with rasterio.open(g["chip_path"]) as cs:
            chip_data, chip_crs, cb, c_res = cs.read(1), cs.crs, cs.bounds, abs(cs.transform.a)
        
        p1 = extract_patch(ortho_path, cb, chip_crs, margin=200.0)
        if p1 is None: continue
        
        chip_up = cv2.resize(chip_data.astype(np.float32), (int(round(chip_data.shape[1]*c_res/PIXEL_SIZE)), int(round(chip_data.shape[0]*c_res/PIXEL_SIZE))), interpolation=cv2.INTER_LINEAR)
        dy, dx, ncc, edge = ncc_match(chip_up, p1)
        if edge or ncc < min_ncc: continue

        off_e, off_n = dx * PIXEL_SIZE, -dy * PIXEL_SIZE
        p2 = extract_patch(ortho_path, cb, chip_crs, margin=30.0, off_e=off_e, off_n=off_n)
        if p2 is None: continue
        
        dy2, dx2, ncc2, edge2 = ncc_match(chip_up, p2)
        if not edge2 and ncc2 >= min_ncc:
            fe, fn = off_e + dx2 * PIXEL_SIZE, off_n - dy2 * PIXEL_SIZE
            if np.hypot(fe - off_e, fn - off_n) < 30.0:
                results.append({"id": g["id"], "east_m": fe, "north_m": fn, "total_m": np.hypot(fe, fn), "ncc": ncc2})

    if len(results) >= 3:
        # Robust MAD filter
        totals = np.array([r["total_m"] for r in results])
        med = np.median(totals)
        mad = np.median(np.abs(totals - med))
        cutoff = med + 3.0 * (1.4826 * mad) if mad > 0 else med + 10.0
        results = [r for r in results if r["total_m"] <= cutoff]

    if not results: return {}
    totals = np.array([r["total_m"] for r in results])
    stats = {"n": len(results), "rmse": np.sqrt(np.mean(totals**2)), "rmse_px": np.sqrt(np.mean(totals**2))/PIXEL_SIZE,
             "ce90": np.percentile(totals, 90), "mean": np.mean(totals)}
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"stats": {"total": stats}, "results": results}, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Verified {stats['n']} GCPs. RMSE: {stats['rmse']:.2f} m")
    return {"ncc": {"stats": {"total": stats}, "results": results}}

def run_verify(ortho_path, gcp_json_path, gcp_chip_dir, output_path, min_ncc=0.4, **kwargs):
    return verify_ncc(ortho_path, gcp_json_path, gcp_chip_dir, output_path, min_ncc=min_ncc)