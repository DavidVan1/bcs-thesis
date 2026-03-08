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

import logging
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.warp import reproject, Resampling, transform as rio_transform
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.coords import BoundingBox

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import SceneConfig

# PhiSat-2 GSD (metres)
PIXEL_SIZE = 4.75
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GCP loading / selection  (shared by A and B)
# ═══════════════════════════════════════════════════════════════════════════

def _load_all_gcps(gcp_json_path: Path,
                   chip_dir: Optional[Path]) -> List[dict]:
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
            if cf is not None and chip_dir is not None:
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
                     ortho_crs,
                     ortho_tf,
                     require_chip: bool = True,
                     ) -> List[dict]:
    """Keep GCPs inside ortho footprint with non-zero data."""
    valid: List[dict] = []
    ortho_crs_obj = CRS.from_user_input(ortho_crs)
    ortho_is_geographic = ortho_crs_obj.is_geographic

    for g in gcps:
        if require_chip and g["chip_path"] is None:
            continue

        if ortho_is_geographic:
            x, y = g["lon"], g["lat"]
        else:
            x_t, y_t = rio_transform(
                "EPSG:4326", ortho_crs_obj, [g["lon"]], [g["lat"]]
            )
            x, y = float(x_t[0]), float(y_t[0])

        ob = ortho_bounds
        if not (ob.left <= x <= ob.right
                and ob.bottom <= y <= ob.top):
            continue
        col = int(round((x - ortho_tf.c) / ortho_tf.a))
        row = int(round((y - ortho_tf.f) / ortho_tf.e))
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
                 ortho_bounds, ortho_data, ortho_crs, ortho_tf,
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
                                ortho_crs,
                                ortho_tf, require_chip=require_chip)

    meta = dict(
        n_json_files=n_json,
        n_total=len(all_gcps),
        n_excluded_holdout=n_excl,
        n_in_ortho=len(in_ortho),
    )

    logger.info("  Loaded %d GCPs from %d JSON file(s)", len(all_gcps), n_json)
    if n_excl:
        logger.info("  Excluded %d near calibration tie points", n_excl)
    logger.info("  Inside ortho with valid data: %d", len(in_ortho))

    return in_ortho, meta


# ═══════════════════════════════════════════════════════════════════════════
# NCC helpers  (used by method B)
# ═══════════════════════════════════════════════════════════════════════════

def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude — removes radiometric differences,
    keeps only structural edges."""
    img = img.astype(np.float32)
    if HAS_CV2:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    else:
        from scipy.ndimage import sobel
        gx = sobel(img, axis=1).astype(np.float32)
        gy = sobel(img, axis=0).astype(np.float32)
    return np.sqrt(gx ** 2 + gy ** 2)


def normalised_cross_correlation(
        template: np.ndarray,
        image: np.ndarray,
) -> Tuple[float, float, float, bool]:
    """
    Sub-pixel NCC of *template* inside *image*.

    Returns ``(dy, dx, ncc_peak, edge_hit)`` — offset of template-match
    centre relative to image centre, in pixels.  *edge_hit* is True when
    the peak sits at the search-window border (unreliable).
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

    edge_hit = (px <= 0 or px >= w - 1 or py <= 0 or py >= h - 1)

    return dy, dx, max_val, edge_hit


def gradient_ncc(
        template: np.ndarray,
        image: np.ndarray,
) -> Tuple[float, float, float, bool]:
    """NCC on Sobel gradient-magnitude images.

    Removes cross-sensor radiometric differences, keeps only structure.
    """
    return normalised_cross_correlation(
        _gradient_magnitude(template),
        _gradient_magnitude(image),
    )


def extract_ortho_patch_utm(
        ortho_path: str,
        chip_bounds,
        chip_crs,
        chip_shape: Tuple[int, int],
        chip_res: float,
        work_res: float = PIXEL_SIZE,
        margin_m: float = 200.0,
        offset_e: float = 0.0,
        offset_n: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Extract a patch from the ortho reprojected to the chip's UTM grid
    at *work_res* (default 4.75 m) with an extra *margin_m* search
    margin (in metres).

    *offset_e* / *offset_n* shift the extraction centre (in metres,
    UTM easting / northing) — used by pass-2 of the two-pass scheme
    to centre the fine search on the coarse estimate.

    Returns float32 2-D array or None.
    """
    dst_left   = chip_bounds.left   - margin_m + offset_e
    dst_bottom = chip_bounds.bottom - margin_m + offset_n
    dst_right  = chip_bounds.right  + margin_m + offset_e
    dst_top    = chip_bounds.top    + margin_m + offset_n

    dst_w = int(round((dst_right - dst_left) / work_res))
    dst_h = int(round((dst_top - dst_bottom) / work_res))
    dst_tf = Affine(work_res, 0, dst_left, 0, -work_res, dst_top)

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


def extract_reference_chip_from_raster(
        reference_path: str,
        center_e: float,
        center_n: float,
        chip_crs,
        chip_shape: Tuple[int, int] = (57, 57),
        chip_res: float = 10.0,
) -> Optional[np.ndarray]:
    """Extract a reference chip from an arbitrary raster in a target UTM grid.

    Used for `reference_source='us_naip'`, where no pre-tiled ESA chips exist.
    """
    h, w = chip_shape
    half_w = 0.5 * w * chip_res
    half_h = 0.5 * h * chip_res

    chip_bounds = BoundingBox(
        left=center_e - half_w,
        bottom=center_n - half_h,
        right=center_e + half_w,
        top=center_n + half_h,
    )
    dst_tf = Affine(chip_res, 0, chip_bounds.left,
                    0, -chip_res, chip_bounds.top)

    with rasterio.open(reference_path) as src:
        n_bands = src.count
        if n_bands >= 3:
            channels = []
            for b in range(1, 4):
                ch = np.zeros((h, w), dtype=np.float32)
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
            out = (0.2989 * channels[0]
                   + 0.5870 * channels[1]
                   + 0.1140 * channels[2])
        else:
            out = np.zeros((h, w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_tf,
                dst_crs=chip_crs,
                resampling=Resampling.bilinear,
            )

    if np.count_nonzero(out) < 0.3 * out.size:
        return None
    return out


def _upsample_chip_to_work_res(
        chip_data: np.ndarray,
        chip_res: float,
        work_res: float = PIXEL_SIZE,
) -> np.ndarray:
    """Resample a chip (typically 57×57 @ 10 m) to *work_res* (4.75 m).

    Uses bilinear interpolation via cv2.resize or scipy.zoom.
    Returns float32 2-D array.
    """
    scale = chip_res / work_res
    new_h = int(round(chip_data.shape[0] * scale))
    new_w = int(round(chip_data.shape[1] * scale))

    chip_f32 = chip_data.astype(np.float32)
    if HAS_CV2:
        return cv2.resize(chip_f32, (new_w, new_h),
                          interpolation=cv2.INTER_LINEAR)
    else:
        from scipy.ndimage import zoom
        return zoom(chip_f32, (scale, scale), order=1).astype(np.float32)


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
    logger.info("\n" + "=" * 72)
    logger.info(f"VERIFICATION RESULTS — {method_label}")
    logger.info("=" * 72)

    logger.info(
        f"  GCPs loaded / holdout-excluded / in ortho: "
        f"{meta['n_total']} / {meta['n_excluded_holdout']} / "
        f"{meta['n_in_ortho']}")
    if "n_skipped" in meta:
        logger.info(
            f"  Skipped / low-quality  : "
            f"{meta.get('n_skipped', 0)} / {meta.get('n_low_quality', 0)}")
    if meta.get("n_inconsistent", 0):
        logger.info(f"  Consistency-gate rejects: {meta['n_inconsistent']}")
    if meta.get("n_outlier", 0):
        logger.info(f"  MAD outliers removed   : {meta['n_outlier']}")
    logger.info(f"  Successfully evaluated : {stats.get('n', 0)}")

    if not stats:
        logger.info("  (no usable results)")
        return

    for label, key in [("Easting  (cross-track)", "easting"),
                       ("Northing (along-track)", "northing")]:
        s = stats[key]
        logger.info(f"\n  {label}:")
        logger.info(f"    Mean  : {s['mean']:+8.2f} m  ({s['mean']/ps:+6.2f} px)")
        logger.info(f"    Std   : {s['std']:8.2f} m  ({s['std']/ps:6.2f} px)")
        logger.info(f"    RMSE  : {s['rmse']:8.2f} m  ({s['rmse_px']:6.2f} px)")

    t = stats["total"]
    logger.info("\n  Total 2-D:")
    logger.info(f"    Mean   : {t['mean']:8.2f} m  ({t['mean']/ps:6.2f} px)")
    logger.info(f"    Median : {t['median']:8.2f} m  ({t['median_px']:6.2f} px)")
    logger.info(f"    RMSE   : {t['rmse']:8.2f} m  ({t['rmse_px']:6.2f} px)")
    logger.info(f"    Max    : {t['max']:8.2f} m     Min : {t['min']:.2f} m")

    if "ncc_mean" in stats:
        logger.info(
            f"\n  NCC  mean={stats['ncc_mean']:.3f}  "
            f"median={stats['ncc_median']:.3f}")
    if "matches_mean" in stats:
        logger.info(
            f"  Matches  mean={stats['matches_mean']:.0f}  "
            f"median={stats['matches_median']:.0f}")

    if stats.get("by_quality"):
        logger.info("")
        for q, qs in stats["by_quality"].items():
            logger.info(
                f"  Q{q}: n={qs['n']:3d}  "
                f"RMSE={qs['rmse']:7.2f} m ({qs['rmse_px']:5.2f} px)  "
                f"mean={qs['mean']:7.2f} m")

    logger.info(f"\n  Pixel reference: {ps} m (PhiSat-2 GSD)")
    logger.info("=" * 72)


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
    logger.info("  Saved → %s", path)


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

    logger.info("\n" + "=" * 72)
    logger.info(f"METHOD A — Position-based (ICP) — scene '{config.name}'")
    logger.info("=" * 72)

    with rasterio.open(ortho_path) as src:
        ob = src.bounds
        ortho_data = src.read()
        ortho_crs = src.crs
        ortho_tf   = src.transform
        ortho_crs  = src.crs

    gcps, meta = _select_gcps(config, ob, ortho_data, ortho_crs, ortho_tf,
                              require_chip=False)

    if not gcps:
        logger.info("  No usable GCPs.")
        return dict(results=[], stats={}, meta=meta)

    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("Method A requires pyproj  (pip install pyproj)")

    logger.info(
        f"\n{'ID':<16s} {'Lon':>11s} {'Lat':>10s} {'Q':>2s} "
        f"{'dE_m':>8s} {'dN_m':>8s} {'Err_m':>8s} {'Err_px':>7s}")
    logger.info("─" * 75)

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

        logger.info(
            f"{g['id']:<16s} {g['lon']:11.6f} {g['lat']:10.6f} "
            f"{g['quality']:2d} "
            f"{east_m:+8.1f} {north_m:+8.1f} {total_m:8.1f} "
            f"{total_m / PIXEL_SIZE:7.2f}")

    stats = _compute_stats(results)
    _print_stats(stats, meta, "Position-based (ICP)")

    out = config.output_dir / "verify_position.json"
    _save_json(out, results, stats, meta, "position")

    return dict(results=results, stats=stats, meta=meta)


# ═══════════════════════════════════════════════════════════════════════════
# METHOD B — Two-pass GCP-chip NCC cross-correlation
# ═══════════════════════════════════════════════════════════════════════════

# Two-pass search parameters
_COARSE_MARGIN_M  = 200.0    # pass-1 search radius (metres)
_FINE_MARGIN_M    = 30.0     # pass-2 search radius, centred on coarse hit
_CONSIST_THRESH_M = 30.0     # max coarse/fine disagreement (metres)


def _single_pass_ncc(
        ortho_path: str,
        chip_data: np.ndarray,
        chip_crs,
        chip_bounds,
        chip_res: float,
        chip_shape: Tuple[int, int],
        work_res: float,
        margin_m: float,
        offset_e: float = 0.0,
        offset_n: float = 0.0,
) -> Tuple[Optional[float], Optional[float], Optional[float],
           bool, bool, Optional[np.ndarray]]:
    """Run one NCC pass.  Returns (dy, dx, ncc, edge_hit, no_data, chip_up).

    *dy/dx* are in work_res pixels (image convention: +y down, +x right).
    *offset_e/n* shift the extraction centre for pass-2.
    *chip_up* is the upsampled chip (returned so pass-2 can reuse it).
    """
    patch = extract_ortho_patch_utm(
        ortho_path, chip_bounds, chip_crs, chip_shape, chip_res,
        work_res=work_res, margin_m=margin_m,
        offset_e=offset_e, offset_n=offset_n,
    )
    if patch is None:
        return None, None, None, False, True, None

    chip_up = _upsample_chip_to_work_res(chip_data, chip_res, work_res)

    try:
        dy, dx, ncc, edge_hit = gradient_ncc(chip_up, patch)
    except Exception:
        return None, None, None, False, True, chip_up

    return dy, dx, ncc, edge_hit, False, chip_up


def verify_ncc(config: SceneConfig,
               min_ncc: float = 0.25,
               reference_source: str = "sentinel") -> dict:
    """
    Two-pass NCC verification with consistency gate.

    Pass 1 (coarse): gradient-NCC at 4.75 m resolution with a wide
    200 m search margin.  Finds the approximate offset.

    Pass 2 (fine): gradient-NCC at 4.75 m with a tight 30 m margin
    **centred on the coarse offset**.  Refines the measurement in a
    region where false locks are unlikely.

    Consistency gate: if the coarse and fine offsets disagree by more
    than 30 m, the GCP is rejected as a probable false lock — the two
    independent measurements don't confirm each other.

    MAD outlier rejection runs **after** the consistency gate as a
    final reporting filter to remove genuinely bad GCP chips.
    """
    ortho_path = str(config.ortho_path)

    logger.info("\n" + "=" * 72)
    logger.info(f"METHOD B — Two-pass NCC ({reference_source}) — scene '{config.name}'")
    logger.info("=" * 72)

    with rasterio.open(ortho_path) as src:
        ob = src.bounds
        ortho_data = src.read()
        ortho_crs = src.crs
        ortho_tf   = src.transform

    require_chip = (reference_source == "sentinel")
    gcps, meta = _select_gcps(config, ob, ortho_data, ortho_crs, ortho_tf,
                              require_chip=require_chip)

    if reference_source not in ("sentinel", "us_naip"):
        raise ValueError("reference_source must be 'sentinel' or 'us_naip'")

    if not gcps:
        logger.info("  No usable GCPs for selected reference source.")
        return dict(results=[], stats={}, meta=meta)

    us_ref_path = None
    if reference_source == "us_naip":
        if config.us_national_ortho_path is None or not config.us_national_ortho_path.exists():
            raise FileNotFoundError(
                "US NAIP reference requested but not found. "
                "Run fetch first or set config.us_national_ortho."
            )
        us_ref_path = str(config.us_national_ortho_path)

    work_res = PIXEL_SIZE

    logger.info(
        f"\n{'ID':<16s} {'Lon':>11s} {'Lat':>10s} {'Q':>2s} "
        f"{'dE_m':>8s} {'dN_m':>8s} {'Err_m':>8s} {'Err_px':>7s} "
        f"{'NCC':>6s} {'gate':>6s}")
    logger.info("─" * 90)

    results: List[dict] = []
    skipped = 0
    low_corr = 0
    inconsistent = 0

    for gcp in gcps:
        gid = gcp["id"]
        if reference_source == "sentinel":
            with rasterio.open(gcp["chip_path"]) as cs:
                chip_data = cs.read(1).astype(np.float32)
                chip_crs  = cs.crs
                chip_bounds = cs.bounds
                chip_res  = abs(cs.transform.a)
                chip_shape = cs.shape
        else:
            chip_res = 10.0
            chip_shape = (57, 57)
            chip_crs = CRS.from_epsg(int(gcp["epsg"]))
            half_w = 0.5 * chip_shape[1] * chip_res
            half_h = 0.5 * chip_shape[0] * chip_res
            chip_bounds = BoundingBox(
                left=gcp["x_utm"] - half_w,
                bottom=gcp["y_utm"] - half_h,
                right=gcp["x_utm"] + half_w,
                top=gcp["y_utm"] + half_h,
            )
            chip_data = extract_reference_chip_from_raster(
                us_ref_path,
                center_e=gcp["x_utm"],
                center_n=gcp["y_utm"],
                chip_crs=chip_crs,
                chip_shape=chip_shape,
                chip_res=chip_res,
            )
            if chip_data is None:
                skipped += 1
                logger.info(
                    f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                    f"{gcp['quality']:2d}  "
                    f"{'— no US reference coverage —':>42s}")
                continue

        # ── Pass 1: coarse (wide search) ─────────────────────────────
        dy1, dx1, ncc1, edge1, no_data1, chip_up = _single_pass_ncc(
            ortho_path, chip_data, chip_crs, chip_bounds,
            chip_res, chip_shape, work_res,
            margin_m=_COARSE_MARGIN_M,
        )

        if no_data1:
            skipped += 1
            logger.info(
                f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                f"{gcp['quality']:2d}  "
                f"{'— insufficient ortho coverage —':>42s}")
            continue

        if edge1:
            skipped += 1
            logger.info(
                f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                f"{gcp['quality']:2d}  "
                f"{'— coarse peak at border':>28s}  NCC={ncc1:.3f}")
            continue

        if ncc1 < min_ncc:
            low_corr += 1
            logger.info(
                f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                f"{gcp['quality']:2d}  "
                f"{'— low coarse NCC:':>20s} {ncc1:.3f}")
            continue

        # Coarse offset in metres (UTM: +east, +north)
        coarse_e =  dx1 * work_res
        coarse_n = -dy1 * work_res

        # ── Pass 2: fine (tight search centred on coarse hit) ────────
        dy2, dx2, ncc2, edge2, no_data2, _ = _single_pass_ncc(
            ortho_path, chip_data, chip_crs, chip_bounds,
            chip_res, chip_shape, work_res,
            margin_m=_FINE_MARGIN_M,
            offset_e=coarse_e,
            offset_n=coarse_n,
        )

        if no_data2 or edge2:
            # Fine pass failed — fall back to coarse-only (mark it)
            east_m, north_m, ncc = coarse_e, coarse_n, ncc1
            gate_label = "coarse"
        else:
            # Fine offset is relative to the shifted centre → add coarse
            fine_e = coarse_e + dx2 * work_res
            fine_n = coarse_n + (-dy2 * work_res)

            # ── Consistency gate ─────────────────────────────────────
            disagree = np.hypot(fine_e - coarse_e, fine_n - coarse_n)
            if disagree > _CONSIST_THRESH_M:
                inconsistent += 1
                logger.info(
                    f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                    f"{gcp['quality']:2d}  "
                    f"— inconsistent: coarse={np.hypot(coarse_e,coarse_n):.1f} m "
                    f"fine={np.hypot(fine_e,fine_n):.1f} m "
                    f"Δ={disagree:.1f} m")
                continue

            east_m, north_m = fine_e, fine_n
            ncc = max(ncc1, ncc2)
            gate_label = "ok"

        if ncc < min_ncc:
            low_corr += 1
            logger.info(
                f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
                f"{gcp['quality']:2d}  "
                f"{'— low fine NCC:':>20s} {ncc:.3f}")
            continue

        total_m = float(np.hypot(east_m, north_m))

        results.append(dict(
            id=gid, lon=gcp["lon"], lat=gcp["lat"],
            alt=gcp["alt"], quality=gcp["quality"],
            east_m=float(east_m), north_m=float(north_m),
            total_m=total_m, ncc=float(ncc),
        ))

        logger.info(
            f"{gid:<16s} {gcp['lon']:11.6f} {gcp['lat']:10.6f} "
            f"{gcp['quality']:2d} "
            f"{east_m:+8.1f} {north_m:+8.1f} {total_m:8.1f} "
            f"{total_m / PIXEL_SIZE:7.2f} {ncc:6.3f} "
            f"{gate_label:>6s}")

    # ── MAD outlier rejection (post-consistency-gate) ────────────────
    # The consistency gate already removed false locks.  MAD now only
    # catches genuine anomalies (e.g. a GCP chip whose ground truth is
    # itself wrong).
    n_outlier = 0
    if len(results) >= 5:
        totals = np.array([r["total_m"] for r in results])
        med = np.median(totals)
        mad = np.median(np.abs(totals - med))
        mad_sigma = 1.4826 * mad
        cutoff = med + 3.0 * mad_sigma
        keep = []
        for r in results:
            if r["total_m"] > cutoff:
                n_outlier += 1
                logger.info(
                    f"  [MAD outlier] {r['id']:<16s} "
                    f"err={r['total_m']:.1f} m  "
                    f"(cutoff={cutoff:.1f} m)")
            else:
                keep.append(r)
        results = keep

    meta.update(n_skipped=skipped, n_low_quality=low_corr,
                n_inconsistent=inconsistent, n_outlier=n_outlier,
                min_ncc=min_ncc, reference_source=reference_source)
    stats = _compute_stats(results)
    _print_stats(stats, meta, f"Two-pass NCC ({reference_source})")

    out = config.output_dir / "verify_ncc.json"
    _save_json(out, results, stats, meta, "ncc_two_pass")

    return dict(results=results, stats=stats, meta=meta)


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════

VERIFY_METHODS = ["all", "position", "ncc"]


def run_verification(config: SceneConfig,
                     method: str = "all",
                     min_ncc: float = 0.25,
                     reference_source: str = "sentinel",
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
    if reference_source == "sentinel":
        missing = config.check_inputs("verify")
    else:
        missing = []
        if config.ortho_path is None or not config.ortho_path.exists():
            missing.append(f"Ortho GeoTIFF: {config.ortho_path}")
        if config.gcp_json_path is None or not config.gcp_json_path.exists():
            missing.append(f"GCP JSON: {config.gcp_json_path}")
        if (config.us_national_ortho_path is None
                or not config.us_national_ortho_path.exists()):
            missing.append(f"US national ortho: {config.us_national_ortho_path}")

    if missing:
        raise FileNotFoundError(
            "Missing files for verify:\n  " + "\n  ".join(missing))

    out: dict = {}

    if method in ("all", "position"):
        out["position"] = verify_position(config)

    if method in ("all", "ncc"):
        out["ncc"] = verify_ncc(config, min_ncc=min_ncc,
                                reference_source=reference_source)

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
