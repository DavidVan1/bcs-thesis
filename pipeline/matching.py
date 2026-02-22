"""
Feature matching between PhiSat-2 and Sentinel-2 imagery.

Handles CRS reprojection, image enhancement, feature matching
(pluggable via :mod:`pipeline.matchers`), RANSAC filtering,
and geo-coordinate extraction from the matched keypoints.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling
import torch

from .config import SceneConfig
from .matchers import get_matcher, list_matchers
from .utils import (
    enhance_for_matching,
    find_sentinel_band,
    load_satellite_image,
    save_tie_points,
)


# ── Sentinel reprojection ──────────────────────────────────────────────

def reproject_sentinel_to_phisat(
    sentinel_ds: rasterio.DatasetReader,
    phisat_ds: rasterio.DatasetReader,
    margin_pixels: int = 512,
) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Reproject Sentinel-2 RGB onto the PhiSat-2 pixel grid (+ margin).
    Returns (reprojected_rgb [H, W, 3], transform).
    """
    print(f"\nReprojecting Sentinel-2 to PhiSat grid "
          f"(+{margin_pixels} px margin)…")

    src_transform = phisat_ds.transform
    src_crs = phisat_ds.crs
    h, w = phisat_ds.height, phisat_ds.width

    new_w = w + 2 * margin_pixels
    new_h = h + 2 * margin_pixels
    dst_transform = (src_transform
                     * rasterio.Affine.translation(-margin_pixels,
                                                   -margin_pixels))

    print(f"  Target grid: {new_w}×{new_h}")

    dest = np.zeros((3, new_h, new_w), dtype=np.uint8)
    bands = [1, 2, 3] if sentinel_ds.count >= 3 else [1]

    for i, band_idx in enumerate(bands):
        idx = i if len(bands) > 1 else 0
        reproject(
            source=rasterio.band(sentinel_ds, band_idx),
            destination=dest[idx],
            src_transform=sentinel_ds.transform,
            src_crs=sentinel_ds.crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.cubic,
        )
        if len(bands) == 1:
            dest[1] = dest[0]
            dest[2] = dest[0]

    return np.transpose(dest, (1, 2, 0)), dst_transform


# ── RANSAC filter ──────────────────────────────────────────────────────

def ransac_filter(kp0: np.ndarray, kp1: np.ndarray,
                  threshold: float = 5.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter matches with RANSAC homography.  Returns inlier arrays."""
    if len(kp0) < 4:
        return kp0, kp1
    H, mask = cv2.findHomography(kp0, kp1, cv2.RANSAC, threshold)
    if mask is not None:
        m = mask.ravel().astype(bool)
        return kp0[m], kp1[m]
    return kp0, kp1


# ── Match visualizer ──────────────────────────────────────────────────

class MatchVisualizer:
    """Side-by-side match visualisation (PhiSat rescaled to Sentinel height)."""

    @staticmethod
    def visualize(image0: np.ndarray, image1: np.ndarray,
                  keypoints0: np.ndarray, keypoints1: np.ndarray,
                  output_path: str, max_matches: int = 100) -> None:
        """Save match visualisation to *output_path*."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        total = len(keypoints0)
        if total == 0:
            print("  No matches to visualise.")
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
        
        # Labels at top
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
        print(f"  Saved match visualisation → {output_path}")


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

    print("=" * 60)
    print(f"MATCHING — scene '{config.name}'  matcher={matcher_name}")
    print("=" * 60)

    # 1. Load images
    phisat_img, phisat_ds = load_satellite_image(str(config.phisat_image_path))

    sentinel_path = find_sentinel_band(str(config.sentinel_dir_path),
                                       config.sentinel_band)
    if not sentinel_path:
        raise FileNotFoundError(
            f"Sentinel band '{config.sentinel_band}' not found "
            f"in {config.sentinel_dir_path}")
    _, sentinel_ds = load_satellite_image(sentinel_path)

    # 2. Reproject Sentinel
    sentinel_aligned, sentinel_tf = reproject_sentinel_to_phisat(
        sentinel_ds, phisat_ds, margin_pixels=config.margin_pixels)

    # 3. Enhance
    print("Enhancing images…")
    phi_enh = enhance_for_matching(phisat_img)
    sen_enh = enhance_for_matching(sentinel_aligned)

    # Save debug images
    cv2.imwrite(str(config.debug_phisat_path),
                cv2.cvtColor(phi_enh, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(config.debug_sentinel_path),
                cv2.cvtColor(sen_enh, cv2.COLOR_RGB2BGR))
    print(f"  Debug images → {config.output_dir}")

    # 4. Match
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = get_matcher(matcher_name, device=device,
                          max_keypoints=config.max_keypoints)
    result = matcher.match(phi_enh, sen_enh)
    kp0, kp1 = result["keypoints0"], result["keypoints1"]
    print(f"Raw matches: {len(kp0)}")

    # 5. RANSAC
    kp0, kp1 = ransac_filter(kp0, kp1)
    print(f"After RANSAC: {len(kp0)}")

    if len(kp0) == 0:
        print("No matches found!")
        phisat_ds.close()
        sentinel_ds.close()
        return []

    # 6. Visualise matches  (include matcher name in filename)
    viz_path = config.output_dir / f"matches_{matcher_name}.png"
    MatchVisualizer.visualize(phi_enh, sen_enh, kp0, kp1, str(viz_path))

    # 7. Geo-coordinates
    tie_points: List[Dict] = []
    for (px, py), (sx, sy) in zip(kp0, kp1):
        lon, lat = sentinel_tf * (float(sx), float(sy))
        tie_points.append({
            "phisat_x": float(px),
            "phisat_y": float(py),
            "lon": float(lon),
            "lat": float(lat),
        })

    # 8. Save
    if config.tie_points_csv:
        save_tie_points(tie_points, str(config.tie_points_path))
        print(f"Saved {len(tie_points)} tie points → {config.tie_points_path}")

    phisat_ds.close()
    sentinel_ds.close()

    return tie_points
