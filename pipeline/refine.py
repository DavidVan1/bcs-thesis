"""
GCP-based affine refinement of orthorectified PhiSat-2 imagery.

After initial verification reveals a systematic bias (due to Sentinel-2
reference geolocation error), this module:

  1. Loads the per-GCP error vectors from verification results.
  2. Rejects outliers via MAD-based 3σ clipping.
  3. Computes a robust mean shift (easting, northing) in metres.
  4. Converts to degrees and applies the shift to the GeoTIFF transform.
  5. Writes a corrected ortho GeoTIFF (overwrites the original).
  6. Re-runs verification to confirm improvement.

Usage:
    python -m pipeline.run sf refine --matcher aliked
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

from .config import SceneConfig

# WGS-84
WGS84_A = 6_378_137.0       # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2


def _metres_per_degree(lat_deg: float) -> Tuple[float, float]:
    """
    Return (m_per_deg_lon, m_per_deg_lat) at the given latitude
    using the WGS-84 ellipsoid radii of curvature.
    """
    lat = np.radians(lat_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    W = np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    # Meridional radius of curvature
    M = WGS84_A * (1.0 - WGS84_E2) / (W ** 3)
    # Prime vertical radius of curvature
    N = WGS84_A / W

    m_per_deg_lat = M * np.radians(1.0)     # ~111,132 m at equator
    m_per_deg_lon = N * cos_lat * np.radians(1.0)  # ~111,320 m at equator

    return m_per_deg_lon, m_per_deg_lat


def _mad_clip(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Return boolean mask of inliers using MAD-based σ-clipping."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-10:
        return np.ones(len(arr), dtype=bool)
    threshold = sigma * 1.4826 * mad  # 1.4826 converts MAD to σ
    return np.abs(arr - med) <= threshold


def compute_shift(results: List[dict],
                  sigma_clip: float = 3.0,
                  min_gcps: int = 3,
                  ) -> Optional[Dict]:
    """
    Compute robust mean shift from GCP verification results.

    Parameters
    ----------
    results : list of dicts
        Each dict has 'east_m', 'north_m', 'lon', 'lat', 'ncc', 'id'.
    sigma_clip : float
        MAD-based sigma clipping threshold.
    min_gcps : int
        Minimum number of GCPs after clipping.

    Returns
    -------
    dict with 'dE_m', 'dN_m', 'dLon_deg', 'dLat_deg',
              'n_used', 'n_rejected', 'rejected_ids'
    or None if insufficient GCPs.
    """
    if len(results) < min_gcps:
        print(f"  Only {len(results)} GCPs — need at least {min_gcps}.")
        return None

    east = np.array([r["east_m"] for r in results])
    north = np.array([r["north_m"] for r in results])
    total = np.array([r["total_m"] for r in results])
    ids = [r["id"] for r in results]

    # Combined outlier rejection: clip on easting, northing, and total
    mask_e = _mad_clip(east, sigma_clip)
    mask_n = _mad_clip(north, sigma_clip)
    mask_t = _mad_clip(total, sigma_clip)
    inlier = mask_e & mask_n & mask_t

    n_rejected = int(np.sum(~inlier))
    rejected = [ids[i] for i in range(len(ids)) if not inlier[i]]

    east_in = east[inlier]
    north_in = north[inlier]

    if len(east_in) < min_gcps:
        print(f"  After clipping: only {len(east_in)} GCPs remain — "
              f"need at least {min_gcps}.")
        return None

    # Robust mean shift
    dE_m = float(np.mean(east_in))
    dN_m = float(np.mean(north_in))

    # Average latitude of GCPs for metre→degree conversion
    lats = np.array([r["lat"] for r in results])[inlier]
    avg_lat = float(np.mean(lats))
    m_per_lon, m_per_lat = _metres_per_degree(avg_lat)

    dLon = dE_m / m_per_lon   # easting metres → longitude degrees
    dLat = dN_m / m_per_lat   # northing metres → latitude degrees

    return {
        "dE_m": dE_m,
        "dN_m": dN_m,
        "dLon_deg": dLon,
        "dLat_deg": dLat,
        "avg_lat_deg": avg_lat,
        "m_per_deg_lon": m_per_lon,
        "m_per_deg_lat": m_per_lat,
        "n_used": int(np.sum(inlier)),
        "n_rejected": n_rejected,
        "rejected_ids": rejected,
    }


def apply_shift_to_geotiff(ortho_path: str, dLon: float, dLat: float) -> None:
    """
    Apply a translation to the GeoTIFF's Affine transform.

    The GCP error (dLon, dLat) measures how far the ortho is displaced
    from truth.  We subtract the shift from the origin to correct.
    """
    with rasterio.open(ortho_path) as src:
        old_tf = src.transform
        data = src.read()
        profile = src.profile.copy()

    # Affine: (pixel_size_x, 0, origin_x, 0, -pixel_size_y, origin_y)
    # Subtract dLon/dLat: the error is measured as ortho-minus-truth,
    # so we correct by moving the ortho origin in the opposite direction.
    new_tf = Affine(
        old_tf.a, old_tf.b, old_tf.c - dLon,
        old_tf.d, old_tf.e, old_tf.f - dLat,
    )

    profile.update(transform=new_tf)
    with rasterio.open(ortho_path, "w", **profile) as dst:
        dst.write(data)

    print(f"  Applied shift: dLon = {dLon:+.8f}°, dLat = {dLat:+.8f}°")
    print(f"  Old origin: ({old_tf.c:.8f}, {old_tf.f:.8f})")
    print(f"  New origin: ({new_tf.c:.8f}, {new_tf.f:.8f})")


def run_refine(config: SceneConfig, min_ncc: float = 0.15) -> Optional[Dict]:
    """
    Run GCP-based affine refinement for a scene.

    1. Load existing verification results (or run verification first).
    2. Compute robust mean shift.
    3. Apply shift to ortho GeoTIFF.
    4. Re-run verification with corrected ortho.

    Returns the re-verification result dict.
    """
    from .verify import run_verification

    print("\n" + "=" * 60)
    print(f"REFINE — GCP-based bias correction — scene '{config.name}'")
    print("=" * 60)

    # ── Step 1: Run fresh verification ──────────────────────────────
    print(f"\n  Running initial verification …\n")
    vr = run_verification(config, min_ncc=min_ncc)
    results = vr["results"]

    if not results:
        print("  No GCP results to compute shift from.")
        return None

    # ── Step 2: Compute robust shift ──────────────────────────────
    print(f"\n  Computing robust shift from {len(results)} GCPs …")
    shift = compute_shift(results, sigma_clip=3.0, min_gcps=3)
    if shift is None:
        return None

    print(f"\n  ── Shift Summary ──")
    print(f"    GCPs used    : {shift['n_used']}")
    print(f"    GCPs rejected: {shift['n_rejected']}  "
          f"{shift['rejected_ids']}")
    print(f"    Easting shift : {shift['dE_m']:+.2f} m  "
          f"({shift['dLon_deg']:+.8f}°)")
    print(f"    Northing shift: {shift['dN_m']:+.2f} m  "
          f"({shift['dLat_deg']:+.8f}°)")

    # ── Step 3: Apply shift to ortho GeoTIFF ──────────────────────
    ortho_path = str(config.ortho_path)
    print(f"\n  Applying shift to {ortho_path} …")
    apply_shift_to_geotiff(ortho_path, shift["dLon_deg"], shift["dLat_deg"])

    # Save shift metadata
    shift_json = config.output_dir / "refine_shift.json"
    with open(shift_json, "w") as f:
        json.dump(shift, f, indent=2)
    print(f"  Saved shift metadata → {shift_json}")

    # ── Step 4: Re-run verification ───────────────────────────────
    print(f"\n  Re-running verification with corrected ortho …\n")
    result = run_verification(config, min_ncc=min_ncc)

    return result
