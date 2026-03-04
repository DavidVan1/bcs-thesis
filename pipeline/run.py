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
import sys
from pathlib import Path

from .config import get_scene_config, list_scenes, SceneConfig
from .matchers import list_matchers


STAGES = ["all", "fetch", "match", "calibrate", "orthorectify", "verify"]


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


def run_verify(config: SceneConfig, method: str = "all") -> None:
    """Run GCP-based verification (position / NCC or both)."""
    from .verify import run_verification
    run_verification(config, method=method)


def run_all(config: SceneConfig, matcher_name: str) -> None:
    """Run the complete pipeline: fetch → match → calibrate → orthorectify → verify."""
    print("\n" + "=" * 70)
    print(f"  PHISAT-2 ORTHORECTIFICATION PIPELINE — scene '{config.name}'")
    print("=" * 70 + "\n")

    run_match(config, matcher_name)
    run_calibrate(config)
    run_orthorectify_stage(config)
    run_verify(config, method="all")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PhiSat-2 Orthorectification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "scene",
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
        choices=list_matchers(),
        help=f"Feature matcher. Default: lightglue. Options: {', '.join(list_matchers())}"
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
        "--verify-method",
        type=str,
        default="all",
        choices=["all", "position", "ncc"],
        help=(
            "Verification method: 'position' (A), 'ncc' (B), "
            "or 'all'. Default: all."
        ),
    )

    args = parser.parse_args()

    if args.list_scenes:
        print("Available scenes:")
        for s in list_scenes():
            print(f"  - {s}")
        sys.exit(0)

    if args.list_matchers:
        print("Available matchers:")
        for m in list_matchers():
            print(f"  - {m}")
        sys.exit(0)

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
            print(
                f"Error: Unknown scene '{args.scene}'.\n"
                f"  Available scenes: {', '.join(list_scenes())}\n"
                f"  To fetch data for a new scene, run:\n"
                f"    python -m pipeline.run {args.scene} fetch "
                f"--phisat-dir phisat/phisat_{args.scene}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Set matcher-specific output paths
    config.set_matcher(args.matcher)

    # Dispatch to the selected stage
    stage = args.stage

    if stage == "all":
        run_all(config, args.matcher)
    elif stage == "fetch":
        no_chips = getattr(args, "no_gcp_chips", False)
        from .fetch import run_fetch
        run_fetch(config, no_gcp_chips=no_chips)
    elif stage == "match":
        run_match(config, args.matcher)
    elif stage == "calibrate":
        run_calibrate(config)
    elif stage == "orthorectify":
        run_orthorectify_stage(config)
    elif stage == "verify":
        run_verify(config, method=args.verify_method)
    else:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
