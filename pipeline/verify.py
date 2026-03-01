"""
GCP-based verification of orthorectified PhiSat-2 imagery.

For each Sentinel-2 GCP chip that falls inside the ortho footprint:
  1. Load the Sentinel-2 GCP chip (57×57 px @ 10 m, in UTM).
  2. Extract the corresponding patch from the ortho GeoTIFF.
  3. Cross-correlate to find the sub-pixel offset.
  4. Convert offset to metres → geometric error.

These GCPs are independent of the LightGlue tie points used for
calibration, so this is genuine out-of-sample validation.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import SceneConfig

# Phi-Sat-2 GSD (metres)
PIXEL_SIZE = 4.75


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalised_cross_correlation(
        template: np.ndarray, image: np.ndarray
) -> Tuple[float, float, float]:
    """
    Sub-pixel NCC of *template* inside *image*.

    Returns (dy, dx, ncc_peak) where dy/dx are the offset of the
    template-match centre relative to the image centre, in pixels.
    """
    if HAS_CV2:
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    else:
        from scipy.signal import correlate2d
        t = template - template.mean()
        i = image - image.mean()
        result = correlate2d(i, t, mode="valid")
        norm = np.sqrt(np.sum(t ** 2) * np.sum(i ** 2)) + 1e-12
        result /= norm

    # Integer peak
    if HAS_CV2:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        px, py = max_loc
    else:
        max_val = result.max()
        iy, ix = np.unravel_index(result.argmax(), result.shape)
        px, py = int(ix), int(iy)

    # Parabolic sub-pixel refinement
    h, w = result.shape
    dx_sub = dy_sub = 0.0
    if 0 < px < w - 1:
        dx_sub = 0.5 * (result[py, px + 1] - result[py, px - 1]) / (
            2 * result[py, px] - result[py, px + 1] - result[py, px - 1] + 1e-12)
    if 0 < py < h - 1:
        dy_sub = 0.5 * (result[py + 1, px] - result[py - 1, px]) / (
            2 * result[py, px] - result[py + 1, px] - result[py - 1, px] + 1e-12)

    expected_x = (w - 1) / 2.0
    expected_y = (h - 1) / 2.0
    dx = (px + dx_sub) - expected_x
    dy = (py + dy_sub) - expected_y

    return dy, dx, max_val


def extract_ortho_patch_utm(
        ortho_path: str,
        chip_bounds,
        chip_crs,
        chip_shape: Tuple[int, int],
        chip_res: float,
        margin_px: int = 20,
) -> Optional[np.ndarray]:
    """
    Extract a patch from the ortho GeoTIFF, reprojected to the same
    UTM grid as the GCP chip, with an extra search margin.
    """
    margin_m = margin_px * chip_res

    dst_left = chip_bounds.left - margin_m
    dst_bottom = chip_bounds.bottom - margin_m
    dst_right = chip_bounds.right + margin_m
    dst_top = chip_bounds.top + margin_m

    dst_w = int(round((dst_right - dst_left) / chip_res))
    dst_h = int(round((dst_top - dst_bottom) / chip_res))
    dst_tf = Affine(chip_res, 0, dst_left, 0, -chip_res, dst_top)

    with rasterio.open(ortho_path) as src:
        dst_array = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_tf,
            dst_crs=chip_crs,
            resampling=Resampling.bilinear,
        )

    if np.count_nonzero(dst_array) < 0.3 * dst_array.size:
        return None
    return dst_array


def _load_gcps(gcp_path: Path, chip_dir: Path,
               ortho_bounds, ortho_data: np.ndarray,
               ortho_transform) -> List[dict]:
    """Select GCPs that fall inside the ortho footprint and have chips."""
    with open(gcp_path) as f:
        db = json.load(f)

    all_gcps = db["GCP_DB"]["GCP"]
    valid = []

    for g in all_gcps:
        gi = g["GCP_Info"]
        lon = float(gi["Longitude"])
        lat = float(gi["Latitude"])
        alt = float(gi["Altimetry"]["#text"])
        qscore = int(g["Quality_Indicators"]["Quality_Score"])

        ob = ortho_bounds
        if not (ob.left <= lon <= ob.right and ob.bottom <= lat <= ob.top):
            continue

        # Resolve chip path
        gri = g.get("GRI_List", {}).get("GRI_Measure", {})
        if isinstance(gri, list):
            gri = gri[0]
        chips = gri.get("Chips", {})
        chip_file = chips.get("Chip_File")
        if isinstance(chip_file, dict):
            chip_file = chip_file.get("Chip_File", chip_file)
        if chip_file is None:
            continue

        chip_path = chip_dir / Path(chip_file).name
        if not chip_path.exists():
            chip_path = chip_dir / f"{g['ID']}_00.TIF"
        if not chip_path.exists():
            continue

        # Check ortho has data at that pixel
        col = int((lon - ortho_transform.c) / ortho_transform.a)
        row = int((lat - ortho_transform.f) / ortho_transform.e)
        if not (0 <= row < ortho_data.shape[1]
                and 0 <= col < ortho_data.shape[2]):
            continue
        if ortho_data[0, row, col] == 0:
            continue

        valid.append(dict(
            id=g["ID"], lon=lon, lat=lat, alt=alt,
            quality=qscore, chip_path=chip_path,
            epsg=int(gi["EPSG"]),
            x_utm=float(gi["X"]), y_utm=float(gi["Y"]),
        ))

    return valid


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_verification(config: SceneConfig, min_ncc: float = 0.3) -> dict:
    """
    Run GCP verification for a scene.

    Returns a dict with keys: 'results', 'stats', 'skipped', 'low_corr'.
    """
    missing = config.check_inputs("verify")
    if missing:
        raise FileNotFoundError(
            "Missing files for verify:\n  " + "\n  ".join(missing))

    ortho_path = str(config.ortho_path)
    gcp_path = config.gcp_json_path
    chip_dir = config.gcp_chip_dir_path

    print("=" * 60)
    print(f"VERIFY — scene '{config.name}'")
    print("=" * 60)

    # Load ortho metadata
    with rasterio.open(ortho_path) as src:
        ob = src.bounds
        ortho_data = src.read()
        ortho_tf = src.transform
    print(f"  Ortho bounds : [{ob.left:.4f},{ob.right:.4f}] × "
          f"[{ob.bottom:.4f},{ob.top:.4f}]")
    print(f"  Ortho shape  : {ortho_data.shape}")

    # Select valid GCPs
    gcps_valid = _load_gcps(gcp_path, chip_dir, ob, ortho_data, ortho_tf)
    print(f"  GCPs with chip + valid ortho data: {len(gcps_valid)}")

    if not gcps_valid:
        print("  No usable GCPs — nothing to verify.")
        return {"results": [], "stats": {}, "skipped": 0, "low_corr": 0}

    # Cross-correlation loop
    print(f"\n{'ID':<16s} {'Lon':>11s} {'Lat':>10s} {'Q':>2s} "
          f"{'dE_m':>8s} {'dN_m':>8s} {'Err_m':>8s} {'Err_px':>7s} "
          f"{'NCC':>6s}")
    print("-" * 90)

    results: List[dict] = []
    skipped = low_corr = 0

    for gcp in gcps_valid:
        with rasterio.open(gcp["chip_path"]) as csrc:
            chip_data = csrc.read(1).astype(np.float32)
            chip_crs = csrc.crs
            chip_bounds = csrc.bounds
            chip_res = abs(csrc.transform.a)
            chip_shape = csrc.shape

        ortho_patch = extract_ortho_patch_utm(
            ortho_path, chip_bounds, chip_crs, chip_shape, chip_res)

        if ortho_patch is None:
            skipped += 1
            print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                  f"{gcp['quality']:2d}  {'— no ortho data —':>40s}")
            continue

        chip_norm = (chip_data - chip_data.mean()).astype(np.float32)
        ortho_norm = (ortho_patch - ortho_patch.mean()).astype(np.float32)

        try:
            dy, dx, ncc = normalised_cross_correlation(chip_norm, ortho_norm)
        except Exception as e:
            skipped += 1
            print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                  f"{gcp['quality']:2d}  {'— NCC failed: ' + str(e):>40s}")
            continue

        if ncc < min_ncc:
            low_corr += 1
            print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                  f"{gcp['quality']:2d}  {'— low NCC: ':>20s}{ncc:.3f}")
            continue

        east_m = dx * chip_res
        north_m = -dy * chip_res  # image y down → north up
        total_m = np.hypot(east_m, north_m)

        results.append(dict(
            id=gcp["id"], lon=gcp["lon"], lat=gcp["lat"],
            alt=gcp["alt"], quality=gcp["quality"],
            east_m=east_m, north_m=north_m, total_m=total_m, ncc=ncc,
        ))

        print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
              f"{gcp['quality']:2d} "
              f"{east_m:+8.1f} {north_m:+8.1f} {total_m:8.1f} "
              f"{total_m / PIXEL_SIZE:7.2f} {ncc:6.3f}")

    # Statistics
    stats = _compute_stats(results)
    _print_stats(stats, len(gcps_valid), skipped, low_corr, min_ncc)

    # Save results JSON
    out_json = config.verification_json_path
    _save_results_json(out_json, results, stats, len(gcps_valid),
                       skipped, low_corr)

    return dict(results=results, stats=stats,
                skipped=skipped, low_corr=low_corr)


# ═══════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to native Python types."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _save_results_json(path: Path, results: List[dict],
                       stats: dict, n_valid: int,
                       skipped: int, low_corr: int) -> None:
    """Write verification results to a JSON file."""
    payload = {
        "n_gcps_valid": n_valid,
        "skipped": skipped,
        "low_correlation": low_corr,
        "n_evaluated": len(results),
        "stats": stats,
        "gcps": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved verification results → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stats(results: List[dict]) -> dict:
    if not results:
        return {}
    east = np.array([r["east_m"] for r in results])
    north = np.array([r["north_m"] for r in results])
    total = np.array([r["total_m"] for r in results])
    ncc = np.array([r["ncc"] for r in results])

    def _block(arr, label):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "rmse_px": float(np.sqrt(np.mean(arr ** 2)) / PIXEL_SIZE),
        }

    stats = {
        "n": len(results),
        "easting": _block(east, "easting"),
        "northing": _block(north, "northing"),
        "total": {
            **_block(total, "total"),
            "median": float(np.median(total)),
            "max": float(np.max(total)),
            "min": float(np.min(total)),
        },
        "ncc_mean": float(np.mean(ncc)),
        "ncc_median": float(np.median(ncc)),
    }

    # Per quality-score breakdown
    qs = {}
    for q in sorted(set(r["quality"] for r in results)):
        sub = [r for r in results if r["quality"] == q]
        t = np.array([r["total_m"] for r in sub])
        n_arr = np.array([r["ncc"] for r in sub])
        qs[q] = {
            "n": len(sub),
            "rmse": float(np.sqrt(np.mean(t ** 2))),
            "rmse_px": float(np.sqrt(np.mean(t ** 2)) / PIXEL_SIZE),
            "ncc_mean": float(np.mean(n_arr)),
        }
    stats["by_quality"] = qs

    return stats


def _print_stats(stats: dict, n_valid: int,
                 skipped: int, low_corr: int,
                 min_ncc: float) -> None:
    print("\n" + "=" * 90)
    print("VERIFICATION STATISTICS")
    print("=" * 90)
    print(f"  GCPs with chip inside ortho : {n_valid}")
    print(f"  Skipped (no data / NCC fail): {skipped}")
    print(f"  Low correlation (< {min_ncc})    : {low_corr}")
    print(f"  Successfully evaluated       : {stats.get('n', 0)}")

    if not stats:
        print("\n  No usable cross-correlation results.")
        return

    ps = PIXEL_SIZE
    for label, key in [("Easting (cross-track)", "easting"),
                       ("Northing (along-track)", "northing")]:
        s = stats[key]
        print(f"\n  ── {label} ──")
        print(f"    Mean error : {s['mean']:+.2f} m  ({s['mean'] / ps:+.2f} px)")
        print(f"    Std dev    : {s['std']:.2f} m  ({s['std'] / ps:.2f} px)")
        print(f"    RMSE       : {s['rmse']:.2f} m  ({s['rmse_px']:.2f} px)")

    t = stats["total"]
    print(f"\n  ── Total 2-D ──")
    print(f"    Mean error : {t['mean']:.2f} m  ({t['mean'] / ps:.2f} px)")
    print(f"    Median     : {t['median']:.2f} m  ({t['median'] / ps:.2f} px)")
    print(f"    RMSE       : {t['rmse']:.2f} m  ({t['rmse_px']:.2f} px)")
    print(f"    Max        : {t['max']:.2f} m  ({t['max'] / ps:.2f} px)")
    print(f"    Min        : {t['min']:.2f} m  ({t['min'] / ps:.2f} px)")

    print(f"\n  Mean NCC     : {stats['ncc_mean']:.3f}")
    print(f"  Median NCC   : {stats['ncc_median']:.3f}")

    for q, qs in stats.get("by_quality", {}).items():
        print(f"\n  Quality={q}: n={qs['n']}, "
              f"RMSE={qs['rmse']:.2f} m ({qs['rmse_px']:.2f} px), "
              f"mean NCC={qs['ncc_mean']:.3f}")

    print(f"\n  Pixel size reference: {ps} m (Phi-Sat 2 GSD)")
    print("=" * 90)
