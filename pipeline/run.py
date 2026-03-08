"""
PhiSat-2 Orthorectification Pipeline — CLI entry point.

Usage:
    python -m pipeline.run <scene> [<stage>] [--matcher NAME]

Examples:
    python -m pipeline.run sf                      # Run all stages
    python -m pipeline.run sf fetch                # Download Sentinel-2, DEM, GCPs
    python -m pipeline.run sf match                # Matching only
    python -m pipeline.run sf match --matcher xoftr
    python -m pipeline.run la calibrate
    python -m pipeline.run sf orthorectify
    python -m pipeline.run sf verify

    # New scene from a PhiSat folder (no config entry needed):
    python -m pipeline.run my_scene fetch --phisat-dir phisat/phisat_my_scene

Available matchers: lightglue, aliked, xoftr, loftr, efficientloftr, roma, mast3r, dust3r
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import get_scene_config, list_scenes, SceneConfig

logger = logging.getLogger(__name__)


STAGES = ["all", "fetch", "match", "calibrate", "orthorectify", "verify"]
MATCHER_CHOICES = [
    "lightglue",
    "aliked",
    "xoftr",
    "loftr",
    "efficientloftr",
    "roma",
    "mast3r",
    "dust3r",
]


def run_fetch_stage(config: SceneConfig) -> None:
    """Download Sentinel-2, DEM and GCPs from Google Earth Engine / Copernicus."""
    from .fetch import run_fetch
    run_fetch(config)


def run_match(config: SceneConfig, matcher_name: str) -> None:
    """Run the feature matching stage."""
    from .matching import run_matching
    run_matching(config, matcher_name=matcher_name)


def run_calibrate(config: SceneConfig) -> None:
    """Run the sensor calibration stage."""
    from .calibration import run_calibration
    run_calibration(config, verbose=True)


def run_orthorectify_stage(config: SceneConfig) -> None:
    """Run the orthorectification stage."""
    from .orthorectify import run_orthorectify
    run_orthorectify(config)


def run_verify(config: SceneConfig,
               method: str = "all",
               reference_source: str = "sentinel") -> None:
    """Run GCP-based verification (position / NCC or both)."""
    from .verify import run_verification
    run_verification(config, method=method,
                     reference_source=reference_source)


def run_all(config: SceneConfig,
            matcher_name: str,
            reference_source: str = "sentinel") -> None:
    """Run the complete pipeline: fetch → match → calibrate → orthorectify → verify."""
    logger.info("\n" + "=" * 70)
    logger.info("  PHISAT-2 ORTHORECTIFICATION PIPELINE — scene '%s'", config.name)
    logger.info("=" * 70 + "\n")

    run_fetch_stage(config)
    run_match(config, matcher_name)
    run_calibrate(config)
    run_orthorectify_stage(config)
    run_verify(config, method="all", reference_source=reference_source)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PhiSat-2 Orthorectification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=str,
        help=f"Scene name. Available: {', '.join(list_scenes())}"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=STAGES,
        help=f"Pipeline stage to run. Default: all. Options: {', '.join(STAGES)}"
    )
    parser.add_argument(
        "--matcher", "-m",
        type=str,
        default="lightglue",
        choices=MATCHER_CHOICES,
        help=f"Feature matcher. Default: lightglue. Options: {', '.join(MATCHER_CHOICES)}"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available scenes and exit."
    )
    parser.add_argument(
        "--list-matchers",
        action="store_true",
        help="List available matchers and exit."
    )
    parser.add_argument(
        "--phisat-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the PhiSat folder relative to project root "
            "(e.g. phisat/phisat_new).  Required for 'fetch' when the scene "
            "is not yet listed in config.py."
        ),
    )
    parser.add_argument(
        "--no-gcp-chips",
        action="store_true",
        help="Skip downloading GCP reference chips during fetch.",
    )
    parser.add_argument(
        "--no-us-mapping",
        action="store_true",
        help="Skip downloading free US national mapping ortho (NAIP) during fetch.",
    )
    parser.add_argument(
        "--verify-method",
        type=str,
        default="all",
        choices=["all", "position", "ncc"],
        help=(
            "Verification method: 'position' (A), 'ncc' (B), "
            "or 'all'. Default: all."
        ),
    )
    parser.add_argument(
        "--reference-source",
        type=str,
        default="sentinel",
        choices=["sentinel", "us_naip"],
        help=(
            "Reference source for NCC verification: 'sentinel' uses ESA GCP "
            "chips, 'us_naip' uses fetched US national ortho patches. "
            "Default: sentinel."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    if args.list_scenes:
        logger.info("Available scenes:")
        for s in list_scenes():
            logger.info("  - %s", s)
        sys.exit(0)

    if args.list_matchers:
        logger.info("Available matchers:")
        for m in MATCHER_CHOICES:
            logger.info("  - %s", m)
        sys.exit(0)

    if not args.scene:
        parser.error("the following arguments are required: scene")

    # ── Load or build scene config ─────────────────────────────────
    from .config import SceneConfig, PROJECT_ROOT

    try:
        config = get_scene_config(args.scene)
    except KeyError:
        # Scene not in config.py — allow fetch to bootstrap it
        if args.stage == "fetch" and args.phisat_dir:
            phisat_dir = args.phisat_dir
            image_rel = "bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff"
            # Auto-detect image file
            from .fetch import _find_phisat_image as _fpi
            tmp = SceneConfig(
                name=args.scene,
                phisat_dir=phisat_dir,
                phisat_image=image_rel,
            )
            _fpi(tmp)
            config = tmp
        else:
            logger.error(
                "Error: Unknown scene '%s'.\n"
                "  Available scenes: %s\n"
                "  To fetch data for a new scene, run:\n"
                "    python -m pipeline.run %s fetch "
                "--phisat-dir phisat/phisat_%s",
                args.scene, ', '.join(list_scenes()),
                args.scene, args.scene,
            )
            sys.exit(1)

    # Set matcher-specific output paths
    config.set_matcher(args.matcher)

    # Dispatch to the selected stage
    stage = args.stage

    if stage == "all":
        run_all(config, args.matcher,
            reference_source=args.reference_source)
    elif stage == "fetch":
        no_chips = getattr(args, "no_gcp_chips", False)
        no_us_mapping = getattr(args, "no_us_mapping", False)
        from .fetch import run_fetch
        run_fetch(config,
                  no_gcp_chips=no_chips,
                  fetch_us_mapping=not no_us_mapping)
    elif stage == "match":
        run_match(config, args.matcher)
    elif stage == "calibrate":
        run_calibrate(config)
    elif stage == "orthorectify":
        run_orthorectify_stage(config)
    elif stage == "verify":
        run_verify(config, method=args.verify_method,
                   reference_source=args.reference_source)
    else:
        logger.error("Unknown stage: %s", stage)
        sys.exit(1)


if __name__ == "__main__":
    main()
