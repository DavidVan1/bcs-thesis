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
    python -m pipeline.run la rpc_fit
    python -m pipeline.run sf orthorectify
    python -m pipeline.run sf verify

    # New scene from a PhiSat folder (no config entry needed):
    python -m pipeline.run my_scene fetch --phisat-dir phisat/phisat_my_scene

Available matchers: lightglue, aliked, xoftr, loftr, efficientloftr, roma, mast3r, dust3r
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .config import get_scene_config, list_scenes, SceneConfig, PROJECT_ROOT
from .profiler import PipelineProfile, profile_stage, _gpu_available

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Stage registry
# ═══════════════════════════════════════════════════════════════════════════

MATCHER_CHOICES: Sequence[str] = (
    "lightglue", "aliked", "xoftr", "loftr",
    "efficientloftr", "roma", "mast3r", "dust3r",
)

# Ordered sequence of stages in their natural pipeline order.
PIPELINE_ORDER: Sequence[str] = (
    "fetch", "match", "calibrate", "rpc_fit", "orthorectify", "verify",
)

# Valid CLI stage names (includes "all").
STAGE_CHOICES: Sequence[str] = ("all", *PIPELINE_ORDER)


# Stages that typically use GPU acceleration
_GPU_STAGES = {"match", "calibrate", "orthorectify"}


@dataclass(frozen=True)
class StageContext:
    """Runtime context passed to every stage runner."""
    config: SceneConfig
    matcher_name: str = "lightglue"
    verify_method: str = "ncc"
    reference_source: str = "sentinel"
    no_gcp_chips: bool = False
    rpc_grid_size: int = 80
    rpc_output: Optional[str] = None
    rpc_json: Optional[str] = None
    profile: bool = False


# Type alias for a stage function:  (StageContext) -> None
StageRunner = Callable[[StageContext], None]

# Registry mapping stage name → runner function
_STAGE_REGISTRY: Dict[str, StageRunner] = {}


def register_stage(name: str) -> Callable[[StageRunner], StageRunner]:
    """Decorator to register a pipeline stage."""
    def _decorator(fn: StageRunner) -> StageRunner:
        _STAGE_REGISTRY[name] = fn
        return fn
    return _decorator


# ── Stage implementations ──────────────────────────────────────────────

@register_stage("fetch")
def _run_fetch(ctx: StageContext) -> None:
    """Download Sentinel-2, DEM and GCPs from Google Earth Engine / Copernicus."""
    from .fetch import run_fetch
    run_fetch(ctx.config.phisat_dir_path,
              no_gcp_chips=ctx.no_gcp_chips)


@register_stage("match")
def _run_match(ctx: StageContext) -> None:
    """Run the feature matching stage."""
    from .matching import run_matching
    sentinel_tiff = ctx.config.phisat_dir_path / "sentinel.tif"
    if not sentinel_tiff.exists():
        raise FileNotFoundError(f"Missing sentinel.tif in {ctx.config.phisat_dir_path}")
    run_matching(
        ctx.config.phisat_image_path,
        sentinel_tiff,
        ctx.config.output_dir,
        ctx.config.tie_points_path,
        matcher_name=ctx.matcher_name,
        margin_pixels=ctx.config.margin_pixels,
        max_keypoints=ctx.config.max_keypoints,
    )


@register_stage("calibrate")
def _run_calibrate(ctx: StageContext) -> None:
    """Run the sensor calibration stage."""
    from .calibration import run_calibration
    run_calibration(
        ctx.config.aocs_path,
        ctx.config.metadata_path,
        ctx.config.tie_points_path,
        ctx.config.dem_path,
        ctx.config.calib_path,
        f=ctx.config.initial_f,
        cx=ctx.config.cx,
        cy=ctx.config.cy,
        verbose=True,
    )


@register_stage("rpc_fit")
def _run_rpc_fit(ctx: StageContext) -> None:
    """Fit RPC model from calibrated rigorous geometry."""
    from .rpc_fit import fit_rpc
    fit_rpc(
        phisat_tiff=ctx.config.phisat_image_path,
        calibration_path=ctx.config.calib_path,
        dem_path=ctx.config.dem_path,
        output_path=Path(ctx.rpc_output),
        aocs_path=ctx.config.aocs_path,
        metadata_path=ctx.config.metadata_path,
        grid_size=ctx.rpc_grid_size,
        f=ctx.config.initial_f,
        cx=ctx.config.cx,
        cy=ctx.config.cy,
    )


@register_stage("orthorectify")
def _run_orthorectify(ctx: StageContext) -> None:
    """Run the orthorectification stage."""
    from .orthorectify import run_orthorectify
    if not ctx.rpc_json:
        raise ValueError("orthorectify requires an RPC JSON path")
    run_orthorectify(
        ctx.config.phisat_image_path,
        ctx.config.dem_path,
        Path(ctx.rpc_json),
        ctx.config.ortho_path,
    )


@register_stage("verify")
def _run_verify(ctx: StageContext) -> None:
    """Run NCC-based GCP verification."""
    from .verify import run_verify
    run_verify(
        ctx.config.ortho_path,
        ctx.config.gcp_json_path,
        ctx.config.gcp_chip_dir_path,
        ctx.config.verification_json_path,
        tie_points_path=ctx.config.tie_points_path,
        method=ctx.verify_method,
        reference_source=ctx.reference_source,
    )


# ── Pipeline orchestrator ──────────────────────────────────────────────

def run_pipeline(ctx: StageContext,
                 stages: Sequence[str] | None = None) -> None:
    """
    Execute one or more pipeline stages in order.

    Parameters
    ----------
    ctx : StageContext
    stages : sequence of stage names, or None for the full pipeline.
    """
    if stages is None:
        stages = PIPELINE_ORDER

    logger.info("\n" + "=" * 70)
    logger.info("  PHISAT-2 ORTHORECTIFICATION PIPELINE — scene '%s'", ctx.config.name)
    logger.info("  Stages: %s", " → ".join(stages))
    logger.info("=" * 70 + "\n")

    has_gpu = _gpu_available() if ctx.profile else False
    pipeline_prof = PipelineProfile(
        scene=ctx.config.name,
        matcher=ctx.matcher_name,
    ) if ctx.profile else None

    for name in stages:
        runner = _STAGE_REGISTRY.get(name)
        if runner is None:
            raise ValueError(f"Unknown stage '{name}'. "
                             f"Available: {', '.join(_STAGE_REGISTRY)}")

        if ctx.profile:
            use_gpu = has_gpu and name in _GPU_STAGES
            with profile_stage(name, use_gpu=use_gpu) as prof:
                runner(ctx)
            pipeline_prof.stages.append(prof)
            logger.info(
                "   %s finished in %.2f s  |  RSS %.0f MB  |  CPU %.0f%%",
                name, prof.wall_time_s, prof.peak_rss_mb, prof.cpu_percent,
            )
        else:
            runner(ctx)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70 + "\n")

    if pipeline_prof is not None:
        pipeline_prof.total_wall_time_s = sum(
            s.wall_time_s for s in pipeline_prof.stages
        )
        logger.info("\n" + pipeline_prof.summary_table() + "\n")
        out_dir = ctx.config.resolve(f"outputs/{ctx.config.name}")
        if out_dir:
            pipeline_prof.save(out_dir / "resource_profile.json")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_config(args: argparse.Namespace) -> SceneConfig:
    """Resolve or bootstrap a SceneConfig from CLI arguments."""
    try:
        return get_scene_config(args.scene)
    except KeyError:
        pass

    # Scene not in scenes.json — allow fetch to bootstrap it
    if args.stage == "fetch" and args.phisat_dir:
        from .fetch import _find_phisat_image
        chosen_image, metadata_json = _find_phisat_image(PROJECT_ROOT / args.phisat_dir)
        if chosen_image is None:
            raise FileNotFoundError(f"No PhiSat TIFF found in {(PROJECT_ROOT / args.phisat_dir) / 'bands'}")
        config = SceneConfig(
            name=args.scene,
            phisat_dir=args.phisat_dir,
            phisat_image=str(chosen_image.relative_to(PROJECT_ROOT / args.phisat_dir)),
            metadata_json=metadata_json.name if metadata_json else None,
        )
        return config

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhiSat-2 Orthorectification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scene", nargs="?", type=str,
        help=f"Scene name. Available: {', '.join(list_scenes())}",
    )
    parser.add_argument(
        "stage", nargs="?", default="all",
        choices=STAGE_CHOICES,
        help=f"Pipeline stage to run. Default: all. Options: {', '.join(STAGE_CHOICES)}",
    )
    parser.add_argument(
        "--matcher", "-m", type=str, default="lightglue",
        choices=MATCHER_CHOICES,
        help=f"Feature matcher. Default: lightglue. Options: {', '.join(MATCHER_CHOICES)}",
    )
    parser.add_argument(
        "--list-scenes", action="store_true",
        help="List available scenes and exit.",
    )
    parser.add_argument(
        "--list-matchers", action="store_true",
        help="List available matchers and exit.",
    )
    parser.add_argument(
        "--phisat-dir", type=str, default=None, metavar="PATH",
        help=(
            "Path to the PhiSat folder relative to project root "
            "(e.g. phisat/phisat_new).  Required for 'fetch' when the scene "
            "is not yet listed in scenes.json."
        ),
    )
    parser.add_argument(
        "--no-gcp-chips", action="store_true",
        help="Skip downloading GCP reference chips during fetch.",
    )
    parser.add_argument(
        "--verify-method", type=str, default="ncc",
        choices=["all", "ncc"],
        help="Verification method: 'ncc'. 'all' is kept as a compatibility alias. Default: ncc.",
    )
    parser.add_argument(
        "--reference-source", type=str, default="sentinel",
        choices=["sentinel"],
        help=(
            "Reference source for NCC verification. Default: sentinel."
        ),
    )
    parser.add_argument(
        "--profile", action="store_true",
        help=(
            "Measure resource usage (wall time, peak RSS, CPU %%, GPU) "
            "for each stage and save to outputs/<scene>/resource_profile.json."
        ),
    )
    parser.add_argument(
        "--rpc-json", type=str, default=None, metavar="PATH",
        help=(
            "Path to RPC JSON (GDAL-style keys). If omitted, defaults to "
            "outputs/<scene>/rpc_<matcher>.json"
        ),
    )
    parser.add_argument(
        "--rpc-grid-size", type=int, default=80,
        help="Sampling grid size per axis used by rpc_fit stage. Default: 80.",
    )
    parser.add_argument(
        "--rpc-output", type=str, default=None, metavar="PATH",
        help=(
            "Output RPC JSON path used by rpc_fit stage. If omitted, defaults to "
            "outputs/<scene>/rpc_<matcher>.json"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── Info queries ───────────────────────────────────────────────
    if args.list_scenes:
        for s in list_scenes():
            logger.info("  %s", s)
        sys.exit(0)

    if args.list_matchers:
        for m in MATCHER_CHOICES:
            logger.info("  %s", m)
        sys.exit(0)

    if not args.scene:
        parser.error("the following arguments are required: scene")

    # ── Build config + context ─────────────────────────────────────
    config = _build_config(args)
    config.set_matcher(args.matcher)

    default_rpc_path = str(config.output_dir / f"rpc_{args.matcher}.json")
    rpc_output = args.rpc_output or default_rpc_path
    rpc_json = args.rpc_json or rpc_output

    ctx = StageContext(
        config=config,
        matcher_name=args.matcher,
        verify_method=args.verify_method,
        reference_source=args.reference_source,
        no_gcp_chips=getattr(args, "no_gcp_chips", False),
        rpc_grid_size=getattr(args, "rpc_grid_size", 80),
        rpc_output=rpc_output,
        rpc_json=rpc_json,
        profile=getattr(args, "profile", False),
    )

    # ── Dispatch ───────────────────────────────────────────────────
    stage = args.stage
    if stage == "all":
        run_pipeline(ctx)
    else:
        run_pipeline(ctx, stages=[stage])


if __name__ == "__main__":
    main()
