"""
Sentinel-2 GCP verification of orthorectified PhiSat-2 imagery.

Two independent verification methods:

  A) **Position-based (ICP)**
     Compare known GCP (lon, lat) against ortho pixel centres.
     Quick sanity check — no cross-correlation needed.

  B) **GCP-chip NCC cross-correlation**
     Match ESA Sentinel-2 L1C reference chips (57×57 @ 10 m, UTM)
     against ortho patches using normalised cross-correlation.

All methods load GCPs from *every* JSON file in the scene's GCP
directory, exclude any GCP too near a calibration tie point, and
report RMSE / mean error in metres and PhiSat-2 pixels (4.75 m).
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

# PhiSat-2 GSD (metres)
PIXEL_SIZE = 4.75


# ═══════════════════════════════════════════════════════════════════════════
# GCP loading / selection  (shared by A and B)
# ═══════════════════════════════════════════════════════════════════════════

def _load_all_gcps(gcp_json_path: Path, chip_dir: Path) -> List[dict]:
    """
    Load GCPs from *every* JSON in the same directory as the configured
    GCP JSON.  De-duplicates by GCP ID.
    """
    gcp_dir = gcp_json_path.parent
    json_files = sorted(gcp_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No GCP JSON files in {gcp_dir}")

    gcps: List[dict] = []
    seen: set = set()

    for jf in json_files:
        with open(jf) as f:
            db = json.load(f)

        for g in db["GCP_DB"]["GCP"]:
            gid = g["ID"]
            if gid in seen:
                continue
            seen.add(gid)

            gi = g["GCP_Info"]

            # Resolve chip file
            gri = g.get("GRI_List", {}).get("GRI_Measure", {})
            if isinstance(gri, list):
                gri = gri[0]
            chips = gri.get("Chips", {})
            cf = chips.get("Chip_File")
            if isinstance(cf, dict):
                cf = cf.get("Chip_File", cf)

            chip_path = None
            if cf is not None:
                chip_path = chip_dir / Path(cf).name
                if not chip_path.exists():
                    chip_path = chip_dir / f"{gid}_00.TIF"
                if not chip_path.exists():
                    chip_path = None

            gcps.append(dict(
                id=gid,
                lon=float(gi["Longitude"]),
                lat=float(gi["Latitude"]),
                alt=float(gi["Altimetry"]["#text"]),
                quality=int(g["Quality_Indicators"]["Quality_Score"]),
                chip_path=chip_path,
                epsg=int(gi["EPSG"]),
                x_utm=float(gi["X"]),
                y_utm=float(gi["Y"]),
            ))

    return gcps


def _filter_to_ortho(gcps: List[dict],
                     ortho_bounds,
                     ortho_data: np.ndarray,
                     ortho_tf,
                     require_chip: bool = True,
                     ) -> List[dict]:
    """Keep GCPs inside ortho footprint with non-zero data."""
    valid: List[dict] = []
    for g in gcps:
        if require_chip and g["chip_path"] is None:
            continue
        ob = ortho_bounds
        if not (ob.left <= g["lon"] <= ob.right
                and ob.bottom <= g["lat"] <= ob.top):
            continue
        col = int((g["lon"] - ortho_tf.c) / ortho_tf.a)
        row = int((g["lat"] - ortho_tf.f) / ortho_tf.e)
        if not (0 <= row < ortho_data.shape[1]
                and 0 <= col < ortho_data.shape[2]):
            continue
        if ortho_data[0, row, col] == 0:
            continue
        g = dict(g)  # shallow copy
        g["ortho_col"] = col
        g["ortho_row"] = row
        valid.append(g)
    return valid


def _check_holdout(gcps: List[dict],
                   tie_points_path: Optional[Path],
                   radius_m: float = 50.0,
                   ) -> Tuple[List[dict], int]:
    """Remove GCPs within *radius_m* of calibration tie points."""
    if tie_points_path is None or not tie_points_path.exists():
        return gcps, 0

    from .utils import load_tie_points
    tps = load_tie_points(str(tie_points_path))
    if not tps:
        return gcps, 0

    tp_ll = np.array([[tp["lon"], tp["lat"]] for tp in tps])
    rad_deg = radius_m / 111_000.0

    holdout: List[dict] = []
    excluded = 0
    for g in gcps:
        d = np.sqrt((tp_ll[:, 0] - g["lon"]) ** 2
                    + (tp_ll[:, 1] - g["lat"]) ** 2)
        if d.min() > rad_deg:
            holdout.append(g)
        else:
            excluded += 1
    return holdout, excluded


def _select_gcps(config: SceneConfig,
                 ortho_bounds, ortho_data, ortho_tf,
                 require_chip: bool = True,
                 ) -> Tuple[List[dict], dict]:
    """
    Full GCP selection pipeline shared by all three methods.

    Returns (selected_gcps, meta_dict).
    """
    gcp_dir = config.gcp_json_path.parent
    n_json = len(list(gcp_dir.glob("*.json")))

    all_gcps = _load_all_gcps(config.gcp_json_path, config.gcp_chip_dir_path)

    holdout, n_excl = _check_holdout(all_gcps, config.tie_points_path)

    in_ortho = _filter_to_ortho(holdout, ortho_bounds, ortho_data,
                                ortho_tf, require_chip=require_chip)

    meta = dict(
        n_json_files=n_json,
        n_total=len(all_gcps),
        n_excluded_holdout=n_excl,
        n_in_ortho=len(in_ortho),
    )

    print(f"  Loaded {len(all_gcps)} GCPs from {n_json} JSON file(s)")
    if n_excl:
        print(f"  Excluded {n_excl} near calibration tie points")
    print(f"  Inside ortho with valid data: {len(in_ortho)}")

    return in_ortho, meta


# ═══════════════════════════════════════════════════════════════════════════
# NCC helpers  (used by method B)
# ═══════════════════════════════════════════════════════════════════════════

def normalised_cross_correlation(
        template: np.ndarray,
        image: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Sub-pixel NCC of *template* inside *image*.

    Returns ``(dy, dx, ncc_peak)`` — offset of template-match centre
    relative to image centre, in pixels.
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
            2 * result[py, px] - result[py, px + 1]
            - result[py, px - 1] + 1e-12)
    if 0 < py < h - 1:
        dy_sub = 0.5 * (result[py + 1, px] - result[py - 1, px]) / (
            2 * result[py, px] - result[py + 1, px]
            - result[py - 1, px] + 1e-12)

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
    Extract a patch from the ortho reprojected to the chip's UTM grid,
    with an extra *margin_px* search margin.  Returns float32 2-D or None.
    """
    margin_m = margin_px * chip_res
    dst_left   = chip_bounds.left   - margin_m
    dst_bottom = chip_bounds.bottom - margin_m
    dst_right  = chip_bounds.right  + margin_m
    dst_top    = chip_bounds.top    + margin_m

    dst_w = int(round((dst_right - dst_left) / chip_res))
    dst_h = int(round((dst_top - dst_bottom) / chip_res))
    dst_tf = Affine(chip_res, 0, dst_left, 0, -chip_res, dst_top)

    with rasterio.open(ortho_path) as src:
        n_bands = src.count
        if n_bands >= 3:
            # Multi-band ortho → luminance
            channels = []
            for b in range(1, 4):
                ch = np.zeros((dst_h, dst_w), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, b),
                    destination=ch,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_tf,
                    dst_crs=chip_crs,
                    resampling=Resampling.bilinear,
                )
                channels.append(ch)
            dst_array = (0.2989 * channels[0]
                         + 0.5870 * channels[1]
                         + 0.1140 * channels[2])
        else:
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


# ═══════════════════════════════════════════════════════════════════════════
# Statistics  (shared)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stats(results: List[dict]) -> dict:
    """Compute mean, RMSE, std for easting / northing / total."""
    if not results:
        return {}

    east  = np.array([r["east_m"]  for r in results])
    north = np.array([r["north_m"] for r in results])
    total = np.array([r["total_m"] for r in results])

    def _axis(arr):
        return {
            "mean":    float(np.mean(arr)),
            "std":     float(np.std(arr)),
            "rmse":    float(np.sqrt(np.mean(arr ** 2))),
            "rmse_px": float(np.sqrt(np.mean(arr ** 2)) / PIXEL_SIZE),
        }

    stats: dict = {
        "n": len(results),
        "easting":  _axis(east),
        "northing": _axis(north),
        "total": {
            **_axis(total),
            "median":    float(np.median(total)),
            "median_px": float(np.median(total) / PIXEL_SIZE),
            "max":       float(np.max(total)),
            "min":       float(np.min(total)),
        },
    }

    # NCC scores if present
    nccs = [r["ncc"] for r in results if "ncc" in r]
    if nccs:
        ncc = np.array(nccs)
        stats["ncc_mean"]   = float(np.mean(ncc))
        stats["ncc_median"] = float(np.median(ncc))

    # Per quality-score breakdown
    by_q: dict = {}
    for q in sorted(set(r["quality"] for r in results)):
        sub = [r for r in results if r["quality"] == q]
        t = np.array([r["total_m"] for r in sub])
        by_q[q] = {
            "n":       len(sub),
            "rmse":    float(np.sqrt(np.mean(t ** 2))),
            "rmse_px": float(np.sqrt(np.mean(t ** 2)) / PIXEL_SIZE),
            "mean":    float(np.mean(t)),
        }
    stats["by_quality"] = by_q

    return stats


def _print_stats(stats: dict, meta: dict, method_label: str) -> None:
    """Pretty-print verification statistics."""
    ps = PIXEL_SIZE
    print("\n" + "=" * 72)
    print(f"VERIFICATION RESULTS — {method_label}")
    print("=" * 72)

    print(f"  GCPs loaded / holdout-excluded / in ortho: "
          f"{meta['n_total']} / {meta['n_excluded_holdout']} / "
          f"{meta['n_in_ortho']}")
    if "n_skipped" in meta:
        print(f"  Skipped / low-quality  : "
              f"{meta.get('n_skipped', 0)} / {meta.get('n_low_quality', 0)}")
    print(f"  Successfully evaluated : {stats.get('n', 0)}")

    if not stats:
        print("  (no usable results)")
        return

    for label, key in [("Easting  (cross-track)", "easting"),
                       ("Northing (along-track)", "northing")]:
        s = stats[key]
        print(f"\n  {label}:")
        print(f"    Mean  : {s['mean']:+8.2f} m  ({s['mean']/ps:+6.2f} px)")
        print(f"    Std   : {s['std']:8.2f} m  ({s['std']/ps:6.2f} px)")
        print(f"    RMSE  : {s['rmse']:8.2f} m  ({s['rmse_px']:6.2f} px)")

    t = stats["total"]
    print(f"\n  Total 2-D:")
    print(f"    Mean   : {t['mean']:8.2f} m  ({t['mean']/ps:6.2f} px)")
    print(f"    Median : {t['median']:8.2f} m  ({t['median_px']:6.2f} px)")
    print(f"    RMSE   : {t['rmse']:8.2f} m  ({t['rmse_px']:6.2f} px)")
    print(f"    Max    : {t['max']:8.2f} m     Min : {t['min']:.2f} m")

    if "ncc_mean" in stats:
        print(f"\n  NCC  mean={stats['ncc_mean']:.3f}  "
              f"median={stats['ncc_median']:.3f}")
    if "matches_mean" in stats:
        print(f"  Matches  mean={stats['matches_mean']:.0f}  "
              f"median={stats['matches_median']:.0f}")

    if stats.get("by_quality"):
        print()
        for q, qs in stats["by_quality"].items():
            print(f"  Q{q}: n={qs['n']:3d}  "
                  f"RMSE={qs['rmse']:7.2f} m ({qs['rmse_px']:5.2f} px)  "
                  f"mean={qs['mean']:7.2f} m")

    print(f"\n  Pixel reference: {ps} m (PhiSat-2 GSD)")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# JSON I/O
# ═══════════════════════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _save_json(path: Path, results: List[dict], stats: dict,
               meta: dict, method: str) -> None:
    payload = {
        "method": method,
        "pixel_size_m": PIXEL_SIZE,
        **meta,
        "stats": stats,
        "gcps": [
            {k: v for k, v in r.items()
             if k not in ("chip_path",)}
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# METHOD A — Position-based (ICP)
# ═══════════════════════════════════════════════════════════════════════════

def verify_position(config: SceneConfig) -> dict:
    """
    Compare known GCP (lon, lat) against ortho pixel centres.

    For each GCP the nearest ortho pixel centre is computed and the
    easting/northing residual (in the GCP's native UTM) is reported.
    """
    ortho_path = str(config.ortho_path)

    print("\n" + "=" * 72)
    print(f"METHOD A — Position-based (ICP) — scene '{config.name}'")
    print("=" * 72)

    with rasterio.open(ortho_path) as src:
        ob = src.bounds
        ortho_data = src.read()
        ortho_tf   = src.transform
        ortho_crs  = src.crs

    gcps, meta = _select_gcps(config, ob, ortho_data, ortho_tf,
                              require_chip=False)

    if not gcps:
        print("  No usable GCPs.")
        return dict(results=[], stats={}, meta=meta)

    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("Method A requires pyproj  (pip install pyproj)")

    print(f"\n{'ID':<16s} {'Lon':>11s} {'Lat':>10s} {'Q':>2s} "
          f"{'dE_m':>8s} {'dN_m':>8s} {'Err_m':>8s} {'Err_px':>7s}")
    print("─" * 75)

    results: List[dict] = []

    for g in gcps:
        # GCP known position in its native UTM
        e_gcp = g["x_utm"]
        n_gcp = g["y_utm"]
        gcp_epsg = g["epsg"]

        # Ortho pixel index for this GCP
        col = g["ortho_col"]
        row = g["ortho_row"]

        # Ortho pixel centre in ortho CRS (lon/lat geographic)
        lon_px = ortho_tf.c + (col + 0.5) * ortho_tf.a
        lat_px = ortho_tf.f + (row + 0.5) * ortho_tf.e

        # Transform ortho pixel centre → GCP's UTM
        xfm = Transformer.from_crs(ortho_crs, f"EPSG:{gcp_epsg}",
                                   always_xy=True)
        e_px, n_px = xfm.transform(lon_px, lat_px)

        east_m  = e_px - e_gcp
        north_m = n_px - n_gcp
        total_m = np.hypot(east_m, north_m)

        results.append(dict(
            id=g["id"], lon=g["lon"], lat=g["lat"],
            alt=g["alt"], quality=g["quality"],
            east_m=float(east_m), north_m=float(north_m),
            total_m=float(total_m),
        ))

        print(f"{g['id']:<16s} {g['lon']:11.6f} {g['lat']:10.6f} "
              f"{g['quality']:2d} "
              f"{east_m:+8.1f} {north_m:+8.1f} {total_m:8.1f} "
              f"{total_m / PIXEL_SIZE:7.2f}")

    stats = _compute_stats(results)
    _print_stats(stats, meta, "Position-based (ICP)")

    out = config.output_dir / "verify_position.json"
    _save_json(out, results, stats, meta, "position")

    return dict(results=results, stats=stats, meta=meta)


# ═══════════════════════════════════════════════════════════════════════════
# METHOD B — GCP-chip NCC cross-correlation
# ═══════════════════════════════════════════════════════════════════════════

def verify_ncc(config: SceneConfig,
               min_ncc: float = 0.15) -> dict:
    """
    Match ESA Sentinel-2 L1C chips against ortho patches using NCC.
    """
    ortho_path = str(config.ortho_path)

    print("\n" + "=" * 72)
    print(f"METHOD B — NCC cross-correlation — scene '{config.name}'")
    print("=" * 72)

    with rasterio.open(ortho_path) as src:
        ob = src.bounds
        ortho_data = src.read()
        ortho_tf   = src.transform

    gcps, meta = _select_gcps(config, ob, ortho_data, ortho_tf,
                              require_chip=True)

    if not gcps:
        print("  No usable GCPs with chips.")
        return dict(results=[], stats={}, meta=meta)

    print(f"\n{'ID':<16s} {'Lon':>11s} {'Lat':>10s} {'Q':>2s} "
          f"{'dE_m':>8s} {'dN_m':>8s} {'Err_m':>8s} {'Err_px':>7s} "
          f"{'NCC':>6s}")
    print("─" * 80)

    results: List[dict] = []
    skipped = low_corr = 0

    for gcp in gcps:
        with rasterio.open(gcp["chip_path"]) as cs:
            chip_data = cs.read(1).astype(np.float32)
            chip_crs  = cs.crs
            chip_bounds = cs.bounds
            chip_res  = abs(cs.transform.a)
            chip_shape = cs.shape

        patch = extract_ortho_patch_utm(
            ortho_path, chip_bounds, chip_crs, chip_shape, chip_res)

        if patch is None:
            skipped += 1
            print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                  f"{gcp['quality']:2d}  "
                  f"{'— insufficient ortho coverage —':>42s}")
            continue

        chip_norm  = (chip_data - chip_data.mean()).astype(np.float32)
        patch_norm = (patch    - patch.mean()).astype(np.float32)

        try:
            dy, dx, ncc = normalised_cross_correlation(chip_norm, patch_norm)
        except Exception as exc:
            skipped += 1
            print(f"{gcp['id']:<16s}  — NCC error: {exc}")
            continue

        if ncc < min_ncc:
            low_corr += 1
            print(f"{gcp['id']:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                  f"{gcp['quality']:2d}  "
                  f"{'— low NCC:':>20s} {ncc:.3f}")
            continue

        east_m  =  dx * chip_res
        north_m = -dy * chip_res
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

    meta.update(n_skipped=skipped, n_low_quality=low_corr, min_ncc=min_ncc)
    stats = _compute_stats(results)
    _print_stats(stats, meta, "NCC cross-correlation")

    out = config.output_dir / "verify_ncc.json"
    _save_json(out, results, stats, meta, "ncc")

    return dict(results=results, stats=stats, meta=meta)


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════

VERIFY_METHODS = ["all", "position", "ncc"]


def run_verification(config: SceneConfig,
                     method: str = "all",
                     min_ncc: float = 0.15,
                     ) -> dict:
    """
    Run GCP verification for a scene.

    Parameters
    ----------
    config : SceneConfig
    method : str
        ``"all"`` runs A + B.  Or pick ``"position"`` or ``"ncc"``.
    min_ncc : float
        NCC threshold for method B.
    """
    missing = config.check_inputs("verify")
    if missing:
        raise FileNotFoundError(
            "Missing files for verify:\n  " + "\n  ".join(missing))

    out: dict = {}

    if method in ("all", "position"):
        out["position"] = verify_position(config)

    if method in ("all", "ncc"):
        out["ncc"] = verify_ncc(config, min_ncc=min_ncc)

    # Also save NCC results under the legacy path
    if method == "all" and "ncc" in out:
        legacy = config.verification_json_path
        ncc_res = out["ncc"]
        _save_json(legacy,
                   ncc_res.get("results", []),
                   ncc_res.get("stats", {}),
                   ncc_res.get("meta", {}),
                   "ncc (legacy verification_results.json)")

    return out
