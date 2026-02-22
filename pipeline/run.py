"""
PhiSat-2 Orthorectification Pipeline — CLI entry point.

Usage:
    python -m pipeline.run <scene> [<stage>] [--matcher NAME]

Examples:
    python -m pipeline.run sf                      # Run all stages
    python -m pipeline.run sf match                # Matching only
    python -m pipeline.run sf match --matcher xoftr
    python -m pipeline.run la calibrate
    python -m pipeline.run sf orthorectify
    python -m pipeline.run sf verify

Available matchers: lightglue, xoftr, loftr, roma, mast3r, dust3r
"""

import argparse
import sys
from pathlib import Path

from .config import get_scene_config, list_scenes, SceneConfig
from .matchers import list_matchers


STAGES = ["all", "match", "calibrate", "orthorectify", "verify"]


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


def run_verify(config: SceneConfig) -> None:
    """Run GCP-based verification."""
    from .verify import run_verification
    run_verification(config)


def run_all(config: SceneConfig, matcher_name: str) -> None:
    """Run the complete pipeline: match → calibrate → orthorectify → verify."""
    print("\n" + "=" * 70)
    print(f"  PHISAT-2 ORTHORECTIFICATION PIPELINE — scene '{config.name}'")
    print("=" * 70 + "\n")

    run_match(config, matcher_name)
    run_calibrate(config)
    run_orthorectify_stage(config)
    run_verify(config)

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

    # Load scene config
    try:
        config = get_scene_config(args.scene)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set matcher-specific output paths
    config.set_matcher(args.matcher)

    # Dispatch to the selected stage
    stage = args.stage

    if stage == "all":
        run_all(config, args.matcher)
    elif stage == "match":
        run_match(config, args.matcher)
    elif stage == "calibrate":
        run_calibrate(config)
    elif stage == "orthorectify":
        run_orthorectify_stage(config)
    elif stage == "verify":
        run_verify(config)
    else:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
