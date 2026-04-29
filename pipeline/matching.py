"""
Feature matching between PhiSat-2 and Sentinel-2 imagery.

Handles CRS reprojection, image enhancement, feature matching
(pluggable via :mod:`pipeline.matchers`), RANSAC filtering,
and geo-coordinate extraction from the matched keypoints.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
from rasterio import Affine
from rasterio.warp import reproject, Resampling, transform_bounds
from pyproj import Transformer, CRS
from rasterio.warp import calculate_default_transform
import torch

from pipeline.matchers import get_matcher
from pipeline.utils import (
    enhance_for_matching,
    load_satellite_image,
    save_tie_points,
)

logger = logging.getLogger(__name__)


def _determine_common_crs(phisat_ds: rasterio.DatasetReader, sentinel_ds: rasterio.DatasetReader) -> CRS:
    """Determine a shared CRS, defaulting to the projected CRS or calculating a UTM zone."""
    if sentinel_ds.crs and sentinel_ds.crs.is_projected:
        return sentinel_ds.crs
    if phisat_ds.crs and phisat_ds.crs.is_projected:
        return phisat_ds.crs
        
    # Fallback to calculating UTM zone from PhiSat center
    phi_cx = 0.5 * (phisat_ds.bounds.left + phisat_ds.bounds.right)
    phi_cy = 0.5 * (phisat_ds.bounds.bottom + phisat_ds.bounds.top)
    to_wgs84 = Transformer.from_crs(phisat_ds.crs, "EPSG:4326", always_xy=True)
    center_lon, center_lat = to_wgs84.transform(phi_cx, phi_cy)
    
    zone = int(np.floor((center_lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    
    return CRS.from_epsg(epsg)


def _rectify_affine(transform: Affine) -> Affine:
    """
    Corrects invalid or severely anisotropic affine transforms.
    """
    col_vec = np.array([transform.a, transform.d], dtype=float)
    row_vec = np.array([transform.b, transform.e], dtype=float)

    # Compute magnitudes (pixel sizes)
    col_mag = float(np.linalg.norm(col_vec))
    row_mag = float(np.linalg.norm(row_vec))

    # 1. Total Degeneracy Check
    if col_mag < 1e-12 and row_mag < 1e-12:
        logger.warning("Degenerate geotransform (scale 0) detected! Applying 4.5m fallback.")
        return Affine.translation(transform.c, transform.f) * Affine.scale(4.5, -4.5)

    # 2. Check Anisotropy & Partial Degeneracy
    max_mag = max(col_mag, row_mag)
    min_mag = min(col_mag, row_mag)
    anisotropy = max_mag / max(min_mag, 1e-12)

    if anisotropy <= 1.5 and min_mag >= 1e-12:
        return transform

    # 3. Rebuild (Transform is broken: highly anisotropic or one axis collapsed)
    logger.info("Rectifying invalid transform (anisotropy %.2f). Rebuilding isotropically.", anisotropy)

    # Anchor to the "healthiest" axis (the one with the larger magnitude)
    if col_mag >= row_mag:
        base_vec = col_vec
        pixel_size = col_mag
        is_col_base = True
    else:
        base_vec = row_vec
        pixel_size = row_mag
        is_col_base = False

    # Normalize the base direction
    base_dir = base_vec / pixel_size

    # Create a forced orthogonal direction to remove shear caused by the collapse
    ortho_dir = np.array([-base_dir[1], base_dir[0]], dtype=float)

    # Reconstruct column and row vectors using the healthy pixel size
    if is_col_base:
        col_new = base_dir * pixel_size
        row_new = ortho_dir * pixel_size
    else:
        row_new = base_dir * pixel_size
        col_new = ortho_dir * pixel_size

    # 4. Preserve Handedness
    cross_orig = col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0]
    cross_new = col_new[0] * row_new[1] - col_new[1] * row_new[0]

    if abs(cross_orig) > 1e-24:
        if cross_orig * cross_new < 0:
            if is_col_base:
                row_new = -row_new
            else:
                col_new = -col_new
    else:
        if cross_new > 0:
            if is_col_base:
                row_new = -row_new
            else:
                col_new = -col_new

    return Affine(
        col_new[0], row_new[0], transform.c,
        col_new[1], row_new[1], transform.f,
    )

def _reproject_image(
    src_img: np.ndarray, 
    src_ds: rasterio.DatasetReader, 
    src_transform: Affine, 
    dst_transform: Affine, 
    target_w: int, 
    target_h: int, 
    dst_crs: CRS
) -> Tuple[np.ndarray, np.ndarray]:
    """Handles the heavy lifting of reprojecting a single image and computing its validity mask."""
    out = np.zeros((3, target_h, target_w), dtype=np.float32)
    src = src_img if src_img.ndim == 3 else np.stack([src_img]*3, axis=-1)
        
    channels = [0, 1, 2] if src.shape[2] >= 3 else [0]
    for i, channel_idx in enumerate(channels):
        dst_idx = i if len(channels) > 1 else 0
        reproject(
            source=src[..., channel_idx].astype(np.float32),
            destination=out[dst_idx],
            src_transform=src_transform,
            src_crs=src_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
            init_dest_nodata=True,
        )
        if len(channels) == 1:
            out[1] = out[0]
            out[2] = out[0]

    valid = np.zeros((target_h, target_w), dtype=np.uint8)
    reproject(
        source=np.ones((src_ds.height, src_ds.width), dtype=np.uint8),
        destination=valid,
        src_transform=src_transform,
        src_crs=src_ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
        init_dest_nodata=True,
    )
            
    out = np.clip(out, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    return out, valid > 0


def create_independent_scaled_grids(
    phisat_img: np.ndarray,
    phisat_ds: rasterio.DatasetReader,
    sentinel_img: np.ndarray,
    sentinel_ds: rasterio.DatasetReader,
    margin_pixels: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Affine, Affine, CRS, Affine]:
    
    logger.info("Projecting images to independent virtual geometries...")

    common_crs = _determine_common_crs(phisat_ds, sentinel_ds)
    
    phisat_src_transform = _rectify_affine(phisat_ds.transform)

    # 2. Calculate Bounds
    phisat_corners = [
        (0.0, 0.0),
        (float(phisat_ds.width), 0.0),
        (float(phisat_ds.width), float(phisat_ds.height)),
        (0.0, float(phisat_ds.height)),
    ]
    phisat_world = [phisat_src_transform * c for c in phisat_corners]
    phi_xs, phi_ys = zip(*phisat_world)
    
    phi_bounds = transform_bounds(
        phisat_ds.crs, common_crs, min(phi_xs), min(phi_ys), max(phi_xs), max(phi_ys), densify_pts=21
    )
    sen_bounds = transform_bounds(
        sentinel_ds.crs, common_crs, *sentinel_ds.bounds, densify_pts=21
    )

    # 4. Calculate Resolution Target
    phi_res_x = abs((phi_bounds[2] - phi_bounds[0]) / max(1, phisat_ds.width))
    phi_res_y = abs((phi_bounds[3] - phi_bounds[1]) / max(1, phisat_ds.height))
    sen_res_x = abs((sen_bounds[2] - sen_bounds[0]) / max(1, sentinel_ds.width))
    sen_res_y = abs((sen_bounds[3] - sen_bounds[1]) / max(1, sentinel_ds.height))

    target_res = max(phi_res_x, phi_res_y, sen_res_x, sen_res_y)
    max_target_res_m = 20.0
    if target_res > max_target_res_m:
        logger.info("Capping coarse target resolution from %.3f to %.3f for matching", target_res, max_target_res_m)
        target_res = max_target_res_m
        
    pad = max(0, margin_pixels) * target_res

    # 5. Define Output Grids
    phi_left, phi_bottom, phi_right, phi_top = phi_bounds
    phi_left -= pad; phi_right += pad
    phi_bottom -= pad; phi_top += pad

    phi_target_w = int(np.ceil((phi_right - phi_left) / target_res))
    phi_target_h = int(np.ceil((phi_top - phi_bottom) / target_res))

    sen_left, sen_bottom, sen_right, sen_top = sen_bounds
    sen_target_w = int(np.ceil((sen_right - sen_left) / target_res))
    sen_target_h = int(np.ceil((sen_top - sen_bottom) / target_res))

    phi_common_transform = from_origin(phi_left, phi_top, target_res, target_res)
    sen_common_transform = from_origin(sen_left, sen_top, target_res, target_res)
    
    logger.info("PhiSat grid: size=%dx%d | Sentinel grid: size=%dx%d", phi_target_w, phi_target_h, sen_target_w, sen_target_h)
    
    # 6. Reproject Both
    phisat_aligned, phisat_valid = _reproject_image(
        phisat_img, phisat_ds, phisat_src_transform, phi_common_transform, phi_target_w, phi_target_h, common_crs
    )
    sentinel_aligned, sentinel_valid = _reproject_image(
        sentinel_img, sentinel_ds, sentinel_ds.transform, sen_common_transform, sen_target_w, sen_target_h, common_crs
    )

    if not np.any(phisat_valid) or not np.any(sentinel_valid):
        raise ValueError("Invalid reprojection (no valid pixels in one or both images)")

    return phisat_aligned, sentinel_aligned, phi_common_transform, sen_common_transform, common_crs, phisat_src_transform



# ── RANSAC filter ──────────────────────────────────────────────────────

def ransac_filter(
    kp0: np.ndarray,
    kp1: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    threshold: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Filter matches with RANSAC homography. Returns inlier arrays (and confidence if provided)."""
    if len(kp0) < 4:
        return kp0, kp1, confidence
    H, mask = cv2.findHomography(kp0, kp1, cv2.RANSAC, threshold,
                                 confidence=0.99999)
    if mask is not None:
        m = mask.ravel().astype(bool)
        if confidence is not None:
            return kp0[m], kp1[m], confidence[m]
        return kp0[m], kp1[m], None
    return kp0, kp1, confidence


# ── Match visualizer ──────────────────────────────────────────────────

def visualize_matches(image0: np.ndarray, image1: np.ndarray,
                     keypoints0: np.ndarray, keypoints1: np.ndarray,
                     output_path: str, max_matches: int = 100) -> None:
    """Save side-by-side match visualisation (PhiSat rescaled to Sentinel height)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = len(keypoints0)
    if total == 0:
        logger.info("  No matches to visualise.")
        return

    if total > max_matches:
        idx = np.random.choice(total, max_matches, replace=False)
        vis0, vis1 = keypoints0[idx], keypoints1[idx]
    else:
        vis0, vis1 = keypoints0, keypoints1

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    scale = h1 / h0
    new_w0, new_h0 = int(w0 * scale), h1

    img0_r = cv2.resize(image0, (new_w0, new_h0),
                         interpolation=cv2.INTER_LINEAR)
    vis0_s = vis0 * scale

    canvas = np.zeros((h1, new_w0 + w1, 3), dtype=np.uint8)
    canvas[:new_h0, :new_w0] = img0_r
    canvas[:h1, new_w0:] = image1

    fig_w = 16
    fig_h = fig_w * h1 / (new_w0 + w1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(canvas)

    c = (144 / 255, 238 / 255, 144 / 255)

    ax.text(20, 40, f"PhiSat-2  {total} matches",
            fontsize=14, color="yellow", ha='left', va='top')
    ax.text(new_w0 + 20, 40, "Sentinel-2",
            fontsize=14, color="yellow", ha='left', va='top')

    for (x0, y0), (x1, y1) in zip(vis0_s, vis1):
        ax.plot([x0, x1 + new_w0], [y0, y1],
                color=c, linewidth=0.8, alpha=0.6)
        ax.scatter(x0, y0, c=[c], s=15, edgecolors="none")
        ax.scatter(x1 + new_w0, y1, c=[c], s=15, edgecolors="none")

    ax.axis("off")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved match visualisation → %s", output_path)



def run_matching(
    phisat_tiff: Path,
    sentinel_tiff: Path,
    output_dir: Path,
    tie_points_path: Path,
    matcher_name: str = "lightglue",
    margin_pixels: int = 512,
    max_keypoints: int = 2048,
    enhance_percentile: float = 98.0,
) -> List[Dict]:
    """
    Run the full matching pipeline for a scene:
      1. Load PhiSat + Sentinel
      2. Reproject Sentinel onto PhiSat grid
      3. Enhance both
      4. Match with the chosen matcher
      5. RANSAC filter
      6. Compute geo-coordinates from Sentinel transform
      7. Save CSV

    Parameters
    ----------
    phisat_tiff : Path
    sentinel_tiff : Path
    output_dir : Path
    tie_points_path : Path
    matcher_name : str
        One of: lightglue, xoftr, loftr, roma, mast3r, dust3r.

    Returns list of tie-point dicts.
    """
    missing = []
    if not phisat_tiff.exists():
        missing.append(f"PhiSat image: {phisat_tiff}")
    if not sentinel_tiff.exists():
        missing.append(f"Sentinel image: {sentinel_tiff}")
    if missing:
        raise FileNotFoundError(
            "Missing files for matching:\n  " + "\n  ".join(missing))

    logger.info("=" * 60)
    logger.info("MATCHING — matcher=%s", matcher_name)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load images
    phisat_img, phisat_ds = load_satellite_image(str(phisat_tiff))
    sentinel_img, sentinel_ds = load_satellite_image(str(sentinel_tiff))

    # Save native (non-reprojected) PhiSat debug image for geometry sanity checks
    cv2.imwrite(
        str(output_dir / "debug_phisat_native.jpg"),
        cv2.cvtColor(phisat_img, cv2.COLOR_RGB2BGR),
    )

    # 2. Build shared virtual geometry and reproject both images
    phisat_aligned, sentinel_aligned, phi_tf, sen_tf, common_crs, phisat_effective_transform = create_independent_scaled_grids(
        phisat_img, phisat_ds,
        sentinel_img, sentinel_ds,
        margin_pixels=margin_pixels
    )

    # Save raw reprojected debug images
    cv2.imwrite(str(output_dir / "debug_phisat.jpg"),
                cv2.cvtColor(phisat_aligned, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "debug_sentinel.jpg"),
                cv2.cvtColor(sentinel_aligned, cv2.COLOR_RGB2BGR))

    # 3. Enhance for matching
    logger.info("Enhancing images...")
    phi_enh = enhance_for_matching(phisat_aligned)
    sen_enh = enhance_for_matching(sentinel_aligned)

    # Save enhanced debug variants
    cv2.imwrite(str(output_dir / "debug_phisat_enh.jpg"),
                cv2.cvtColor(phi_enh, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "debug_sentinel_enh.jpg"),
                cv2.cvtColor(sen_enh, cv2.COLOR_RGB2BGR))
    logger.info("  Debug images → %s", output_dir)

    # 4. Match (using a sliding window over the massive Sentinel image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = get_matcher(matcher_name, device=device,
                          max_keypoints=max_keypoints)
    
    phi_h, phi_w = phi_enh.shape[:2]
    sen_h, sen_w = sen_enh.shape[:2]
    
    # We stride across the Sentinel image in chunks roughly the size of the PhiSat image.
    # This guarantees the neural network sees exactly the same scale
    step_y = max(1, int(phi_h * 0.75))
    step_x = max(1, int(phi_w * 0.75))
    
    all_kp0 = []
    all_kp1 = []
    all_conf = []
    
    logger.info("Matching via sliding window (window %dx%d, sentinel %dx%d)...", phi_w, phi_h, sen_w, sen_h)
    
    for y_start in range(0, sen_h, step_y):
        for x_start in range(0, sen_w, step_x):
            y_end = min(sen_h, y_start + phi_h)
            x_end = min(sen_w, x_start + phi_w)
            
            # If the crop at the edge is too small, shift it back to maintain size
            # so the matcher doesn't act weirdly on tiny strips.
            if y_end - y_start < phi_h and y_start > 0:
                y_start = max(0, y_end - phi_h)
            if x_end - x_start < phi_w and x_start > 0:
                x_start = max(0, x_end - phi_w)
                
            sen_crop = sen_enh[y_start:y_end, x_start:x_end]
            
            # Only match if the crop has legitimate valid data (not entirely black/empty padding)
            if np.mean(sen_crop) > 5.0:
                res = matcher.match(phi_enh, sen_crop)
                
                if len(res["keypoints0"]) > 0:
                    all_kp0.append(res["keypoints0"])
                    all_conf.append(res["confidence"])
                    
                    # Shift Sentinel keypoints back to full-image coordinates
                    shifted_kp1 = res["keypoints1"] + np.array([[x_start, y_start]])
                    all_kp1.append(shifted_kp1)
    
    if all_kp0:
        kp0 = np.concatenate(all_kp0, axis=0)
        kp1 = np.concatenate(all_kp1, axis=0)
        conf = np.concatenate(all_conf, axis=0)
    else:
        kp0, kp1, conf = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        
    logger.info("Raw matches (accumulated): %d", len(kp0))

    # 5. RANSAC
    kp0, kp1, conf = ransac_filter(kp0, kp1, conf)
    logger.info("After RANSAC: %d", len(kp0))

    max_saved_matches = 5000
    if len(kp0) > max_saved_matches:
        best_idx = np.argsort(conf)[-max_saved_matches:]
        best_idx = best_idx[np.argsort(conf[best_idx])[::-1]]
        kp0 = kp0[best_idx]
        kp1 = kp1[best_idx]
        conf = conf[best_idx]
        logger.info("Keeping top %d matches by confidence", max_saved_matches)

    if len(kp0) == 0:
        logger.warning("No matches found!")
        phisat_ds.close()
        sentinel_ds.close()
        return []

    # 6. Visualise matches  (include matcher name in filename)
    viz_path = output_dir / f"matches_{matcher_name}.png"
    visualize_matches(phi_enh, sen_enh, kp0, kp1, str(viz_path))

    # 7. Geo-coordinates
    phi_transform_inv = ~phisat_effective_transform

    common_to_phi = Transformer.from_crs(common_crs, phisat_ds.crs, always_xy=True)
    common_to_wgs84 = Transformer.from_crs(common_crs, "EPSG:4326", always_xy=True)

    tie_points: List[Dict] = []
    for (px, py), (sx, sy) in zip(kp0, kp1):
        # Coordinates in aligned grid map identically through the master transform
        common_x_phi, common_y_phi = phi_tf * (float(px), float(py))
        phi_x, phi_y = common_to_phi.transform(common_x_phi, common_y_phi)
        orig_px, orig_py = phi_transform_inv * (phi_x, phi_y)

        common_x_sen, common_y_sen = sen_tf * (float(sx), float(sy))
        lon, lat = common_to_wgs84.transform(common_x_sen, common_y_sen)

        tie_points.append({
            "phisat_x": float(orig_px),
            "phisat_y": float(orig_py),
            "lon": float(lon),
            "lat": float(lat),
        })

    # 8. Save
    save_tie_points(tie_points, str(tie_points_path))
    logger.info("Saved %d tie points → %s", len(tie_points), tie_points_path)

    phisat_ds.close()
    sentinel_ds.close()

    return tie_points
