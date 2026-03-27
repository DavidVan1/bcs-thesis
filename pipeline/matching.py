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
import torch

from .config import SceneConfig
from .matchers import get_matcher, list_matchers
from .utils import (
    enhance_for_matching,
    find_sentinel_band,
    load_satellite_image,
    save_tie_points,
)

logger = logging.getLogger(__name__)


# ── Sentinel reprojection ──────────────────────────────────────────────

def create_scaled_intersection_grid(
    phisat_img: np.ndarray,
    phisat_ds: rasterio.DatasetReader,
    sentinel_img: np.ndarray,
    sentinel_ds: rasterio.DatasetReader,
    margin_pixels: int = 0,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine, rasterio.crs.CRS, rasterio.Affine]:
    """
    Reproject both images onto one shared virtual geometry for local matching:
    - same CRS
    - same pixel grid
    - same overlap extent
    - same target resolution (coarser/native-safe)
    """
    logger.info("Projecting both images to shared virtual geometry...")

    if sentinel_ds.crs and sentinel_ds.crs.is_projected:
        common_crs = sentinel_ds.crs
    elif phisat_ds.crs and phisat_ds.crs.is_projected:
        common_crs = phisat_ds.crs
    else:
        phi_cx = 0.5 * (phisat_ds.bounds.left + phisat_ds.bounds.right)
        phi_cy = 0.5 * (phisat_ds.bounds.bottom + phisat_ds.bounds.top)
        to_wgs84 = Transformer.from_crs(phisat_ds.crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs84.transform(phi_cx, phi_cy)
        zone = int(np.floor((center_lon + 180.0) / 6.0) + 1)
        zone = max(1, min(60, zone))
        epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
        common_crs = CRS.from_epsg(epsg)

    phisat_src_transform = phisat_ds.transform
    col_vec = np.array([phisat_src_transform.a, phisat_src_transform.d], dtype=float)
    row_vec = np.array([phisat_src_transform.b, phisat_src_transform.e], dtype=float)
    col_mag = float(np.linalg.norm(col_vec))
    row_mag = float(np.linalg.norm(row_vec))
    anisotropy = max(col_mag, row_mag) / max(min(col_mag, row_mag), 1e-12)
    if anisotropy > 1.5:
        if row_mag < 1e-12:
            row_dir = np.array([-col_vec[1], col_vec[0]], dtype=float)
            row_dir /= max(float(np.linalg.norm(row_dir)), 1e-12)
        else:
            row_dir = row_vec / row_mag
        row_new = row_dir * col_mag
        cross = col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0]
        cross_new = col_vec[0] * row_new[1] - col_vec[1] * row_new[0]
        if cross * cross_new < 0:
            row_new = -row_new
        phisat_src_transform = Affine(
            col_vec[0], row_new[0], phisat_src_transform.c,
            col_vec[1], row_new[1], phisat_src_transform.f,
        )
        logger.info(
            "Applied PhiSat isotropic transform correction for matching (anisotropy %.2f)",
            anisotropy,
        )

    phisat_corners = [
        (0.0, 0.0),
        (float(phisat_ds.width), 0.0),
        (float(phisat_ds.width), float(phisat_ds.height)),
        (0.0, float(phisat_ds.height)),
    ]
    phisat_world = [phisat_src_transform * c for c in phisat_corners]
    phi_xs, phi_ys = zip(*phisat_world)
    phi_bounds = transform_bounds(
        phisat_ds.crs,
        common_crs,
        min(phi_xs), min(phi_ys), max(phi_xs), max(phi_ys),
        densify_pts=21,
    )
    sen_bounds = transform_bounds(
        sentinel_ds.crs,
        common_crs,
        *sentinel_ds.bounds,
        densify_pts=21,
    )

    inter_left = max(phi_bounds[0], sen_bounds[0])
    inter_bottom = max(phi_bounds[1], sen_bounds[1])
    inter_right = min(phi_bounds[2], sen_bounds[2])
    inter_top = min(phi_bounds[3], sen_bounds[3])

    if inter_right <= inter_left or inter_top <= inter_bottom:
        raise ValueError("No overlap between PhiSat and Sentinel after CRS harmonization")

    union_left = min(phi_bounds[0], sen_bounds[0])
    union_bottom = min(phi_bounds[1], sen_bounds[1])
    union_right = max(phi_bounds[2], sen_bounds[2])
    union_top = max(phi_bounds[3], sen_bounds[3])

    phi_res_x = abs((phi_bounds[2] - phi_bounds[0]) / max(1, phisat_ds.width))
    phi_res_y = abs((phi_bounds[3] - phi_bounds[1]) / max(1, phisat_ds.height))
    sen_res_x = abs((sen_bounds[2] - sen_bounds[0]) / max(1, sentinel_ds.width))
    sen_res_y = abs((sen_bounds[3] - sen_bounds[1]) / max(1, sentinel_ds.height))

    target_res = max(phi_res_x, phi_res_y, sen_res_x, sen_res_y)
    max_target_res_m = 20.0
    if target_res > max_target_res_m:
        logger.info(
            "Capping coarse target resolution from %.3f to %.3f for matching",
            target_res,
            max_target_res_m,
        )
        target_res = max_target_res_m
    pad = max(0, margin_pixels) * target_res

    # Keep full PhiSat footprint (do not cut it off), optionally extend with
    # a matching margin around the overlap area, clipped to union bounds.
    left = min(phi_bounds[0], inter_left - pad)
    right = max(phi_bounds[2], inter_right + pad)
    bottom = min(phi_bounds[1], inter_bottom - pad)
    top = max(phi_bounds[3], inter_top + pad)

    left = max(union_left, left)
    right = min(union_right, right)
    bottom = max(union_bottom, bottom)
    top = min(union_top, top)

    target_w = int(np.ceil((right - left) / target_res))
    target_h = int(np.ceil((top - bottom) / target_res))
    if target_w < 1 or target_h < 1:
        raise ValueError("Invalid target grid size from shared geometry")

    common_transform = from_origin(left, top, target_res, target_res)
    logger.info(
        "Shared grid: CRS=%s, res=%.3f, size=%dx%d",
        common_crs,
        target_res,
        target_w,
        target_h,
    )
    
    def reproject_img(src_img, src_ds, src_transform, dst_transform, target_w, target_h, dst_crs):
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

    # Project both to the new 10m shared grid
    phisat_aligned, phisat_valid = reproject_img(
        phisat_img, phisat_ds, phisat_src_transform, common_transform, target_w, target_h, common_crs
    )
    sentinel_aligned, sentinel_valid = reproject_img(
        sentinel_img, sentinel_ds, sentinel_ds.transform, common_transform, target_w, target_h, common_crs
    )

    overlap_valid = phisat_valid & sentinel_valid
    if not np.any(overlap_valid):
        raise ValueError("No mutual valid pixels after reprojection")

    overlap_pixels = int(np.count_nonzero(overlap_valid))
    phi_pixels = int(np.count_nonzero(phisat_valid))
    overlap_ratio = overlap_pixels / max(1, phi_pixels)
    logger.info(
        "Keeping full PhiSat canvas: size=%dx%d (overlap with Sentinel: %.1f%% of PhiSat area)",
        target_w,
        target_h,
        100.0 * overlap_ratio,
    )

    return phisat_aligned, sentinel_aligned, common_transform, common_crs, phisat_src_transform

# ── RANSAC filter ──────────────────────────────────────────────────────

def ransac_filter(kp0: np.ndarray, kp1: np.ndarray,
                  threshold: float = 8.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter matches with RANSAC homography.  Returns inlier arrays."""
    if len(kp0) < 4:
        return kp0, kp1
    H, mask = cv2.findHomography(kp0, kp1, cv2.RANSAC, threshold,
                                 confidence=0.99999)
    if mask is not None:
        m = mask.ravel().astype(bool)
        return kp0[m], kp1[m]
    return kp0, kp1


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


# ── Full matching pipeline ─────────────────────────────────────────────

def run_matching(config: SceneConfig,
                 matcher_name: str = "lightglue") -> List[Dict]:
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
    config : SceneConfig
    matcher_name : str
        One of: lightglue, xoftr, loftr, roma, mast3r, dust3r.

    Returns list of tie-point dicts.
    """
    missing = config.check_inputs("matching")
    if missing:
        raise FileNotFoundError(
            "Missing files for matching:\n  " + "\n  ".join(missing))

    logger.info("=" * 60)
    logger.info("MATCHING — scene '%s'  matcher=%s", config.name, matcher_name)
    logger.info("=" * 60)

    config.ensure_output_dir()

    # 1. Load images
    phisat_img, phisat_ds = load_satellite_image(str(config.phisat_image_path))

    sentinel_path = find_sentinel_band(str(config.sentinel_dir_path),
                                       config.sentinel_band)
    if not sentinel_path:
        raise FileNotFoundError(
            f"Sentinel band '{config.sentinel_band}' not found "
            f"in {config.sentinel_dir_path}")
    sentinel_img, sentinel_ds = load_satellite_image(sentinel_path)

    # Save native (non-reprojected) PhiSat debug image for geometry sanity checks
    cv2.imwrite(
        str(config.output_dir / "debug_phisat_native.jpg"),
        cv2.cvtColor(phisat_img, cv2.COLOR_RGB2BGR),
    )

    # 2. Build shared virtual geometry and reproject both images
    phisat_aligned, sentinel_aligned, common_tf, common_crs, phisat_effective_transform = create_scaled_intersection_grid(
        phisat_img, phisat_ds,
        sentinel_img, sentinel_ds,
        margin_pixels=config.margin_pixels
    )

    # Save raw reprojected debug images
    cv2.imwrite(str(config.debug_phisat_path),
                cv2.cvtColor(phisat_aligned, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(config.debug_sentinel_path),
                cv2.cvtColor(sentinel_aligned, cv2.COLOR_RGB2BGR))

    # 3. Enhance for matching
    logger.info("Enhancing images...")
    phi_enh = enhance_for_matching(phisat_aligned)
    sen_enh = enhance_for_matching(sentinel_aligned)

    # Save enhanced debug variants
    cv2.imwrite(str(config.output_dir / "debug_phisat_enh.jpg"),
                cv2.cvtColor(phi_enh, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(config.output_dir / "debug_sentinel_enh.jpg"),
                cv2.cvtColor(sen_enh, cv2.COLOR_RGB2BGR))
    logger.info("  Debug images → %s", config.output_dir)

    # 4. Match
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = get_matcher(matcher_name, device=device,
                          max_keypoints=config.max_keypoints)
    result = matcher.match(phi_enh, sen_enh)
    kp0, kp1 = result["keypoints0"], result["keypoints1"]
    logger.info("Raw matches: %d", len(kp0))

    # 5. RANSAC
    kp0, kp1 = ransac_filter(kp0, kp1)
    logger.info("After RANSAC: %d", len(kp0))

    if len(kp0) == 0:
        logger.warning("No matches found!")
        phisat_ds.close()
        sentinel_ds.close()
        return []

    # 6. Visualise matches  (include matcher name in filename)
    viz_path = config.output_dir / f"matches_{matcher_name}.png"
    visualize_matches(phi_enh, sen_enh, kp0, kp1, str(viz_path))

    # 7. Geo-coordinates
    phi_transform_inv = ~phisat_effective_transform

    common_to_phi = Transformer.from_crs(common_crs, phisat_ds.crs, always_xy=True)
    common_to_wgs84 = Transformer.from_crs(common_crs, "EPSG:4326", always_xy=True)

    tie_points: List[Dict] = []
    for (px, py), (sx, sy) in zip(kp0, kp1):
        # Coordinates in aligned grid map identically through the master transform
        common_x_phi, common_y_phi = common_tf * (float(px), float(py))
        phi_x, phi_y = common_to_phi.transform(common_x_phi, common_y_phi)
        orig_px, orig_py = phi_transform_inv * (phi_x, phi_y)

        common_x_sen, common_y_sen = common_tf * (float(sx), float(sy))
        lon, lat = common_to_wgs84.transform(common_x_sen, common_y_sen)

        tie_points.append({
            "phisat_x": float(orig_px),
            "phisat_y": float(orig_py),
            "lon": float(lon),
            "lat": float(lat),
        })

    # 8. Save
    if config.tie_points_csv:
        save_tie_points(tie_points, str(config.tie_points_path))
        logger.info("Saved %d tie points → %s", len(tie_points), config.tie_points_path)

    phisat_ds.close()
    sentinel_ds.close()

    return tie_points
