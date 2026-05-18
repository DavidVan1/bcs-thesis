from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import json
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict

from pipeline.evaluation.orthorectify import run_orthorectify
from pipeline.evaluation.validation import run_validation
from pipeline.utils import configure_pipeline_file_logger, log_pipeline_stage

def worker_task(
    scene_dir: Path, 
    rpc_dir: Path, 
    output_root: Path, 
    matcher: str, 
    min_ncc: float, 
    overwrite: bool
) -> Dict[str, object]:
    """
    Standard worker task for orthorectification and verification.
    Runs on a single CPU core via ProcessPoolExecutor.
    """
    scene_name = scene_dir.name
    print(f"[ RUN  ] {scene_name}", flush=True)
    
    try:
        pixel_size_m = 4.75
        # 1. Directory & Path Setup
        out_dir = output_root / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Locate input files
        bands_dir = scene_dir / "bands"
        # Find RGB TIFF
        rgb_tiff = next(bands_dir.glob("*_RGB.tiff"), None)
        if not rgb_tiff:
            # Fallback for different naming conventions
            tiffs = [f for f in scene_dir.glob("*.tif*") if "sentinel" not in f.name and "dem" not in f.name]
            if not tiffs:
                raise FileNotFoundError(f"No valid TIFF found in {scene_dir}")
            rgb_tiff = tiffs[0]

        dem_path = scene_dir / "dem.tif"
        rpc_txt_path = rpc_dir / f"{scene_name}_RPC.txt"
        ortho_path = out_dir / f"ortho_reference_{matcher}.tif"

        if not rpc_txt_path.exists():
            raise FileNotFoundError(f"RPC file missing: {rpc_txt_path}")

        # 2. Orthorectification Stage
        # Logic: If overwrite is TRUE, always run. If FALSE, only run if file is missing.
        if overwrite or not ortho_path.exists():
            run_orthorectify(
                phisat_tiff=rgb_tiff,
                dem_path=dem_path,
                rpc_path=rpc_txt_path,
                output_path=ortho_path,
            )

        # 3. Verification Stage
        gcp_root = scene_dir / "sentinel_gri"
        gcp_json = next(gcp_root.glob("*.json"), None)
        gcp_chips = gcp_root / "L1C_chips"
        
        if not gcp_json:
            raise FileNotFoundError(f"No GCP JSON found in {gcp_root}")

        verify_out = out_dir / f"verification_ncc_{matcher}.json"

        # run_validation returns the metrics dict
        verify_result = run_validation(
            ortho_path=ortho_path,
            gcp_json_path=gcp_json,
            gcp_chip_dir=gcp_chips,
            output_path=verify_out,
            min_ncc=min_ncc,
            reference_source="sentinel",
        )

        # 4. Extract metrics for logging/summary
        ncc_data = verify_result.get("ncc", {})
        stats = ncc_data.get("stats", {}).get("total", {})

        rmse = stats.get("rmse")
        ce90 = stats.get("ce90")
        n_gcp = stats.get("n", 0)

        rmse_px = (rmse / pixel_size_m) if rmse is not None else None
        ce90_px = (ce90 / pixel_size_m) if ce90 is not None else None

        if n_gcp == 0 or rmse is None:
            print(f"[ SKIP ] {scene_name} - No valid GCP matches found.", flush=True)
            log_pipeline_stage(
                scene=scene_name, 
                matcher=matcher, 
                stage="verify", 
                status="SKIP", 
                reason="zero_gcp_matches"
            )
            return {
                "scene": scene_name, "status": "skipped", 
                "rmse": None, "ce90": None, "error": "zero_gcp_matches"
            }

        # Success Log
        status_msg = f"[  OK  ] {scene_name}  rmse={rmse:.2f}m  ce90={ce90:.2f}m"
        print(status_msg, flush=True)
        
        # Log to shared pipeline database/file
        log_pipeline_stage(
            scene=scene_name, 
            matcher=matcher, 
            stage="verify", 
            status="OK", 
            metrics={"rmse": rmse, "ce90": ce90}
        )

        return {
            "scene": scene_name, "status": "ok", 
            "rmse": rmse, "ce90": ce90,
            "rmse_px": rmse_px, "ce90_px": ce90_px,
            "error": ""
        }

    except Exception as e:
        error_msg = f"[ FAIL ] {scene_name}  reason={str(e)}"
        print(error_msg, flush=True)
        log_pipeline_stage(
            scene=scene_name, 
            matcher=matcher, 
            stage="verify", 
            status="FAIL", 
            reason=str(e)
        )
        return {
            "scene": scene_name, "status": "fail",
            "rmse": None, "ce90": None,
            "rmse_px": None, "ce90_px": None,
            "error": str(e)
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Orthorectification and Verification Pipeline."
    )
    parser.add_argument("--dataset", default="data/scenes", help="Input scenes directory")
    parser.add_argument("--output-root", default="data/02_reference", help="Root for orthos and metrics")
    parser.add_argument("--rpc-dir", default=None, help="Dir with _RPC.txt (defaults to data/reference_rpc/<matcher>)")
    parser.add_argument("--matcher", default="lightglue", help="Matcher tag used for filenames")
    parser.add_argument("--scene", default=None, help="Optional: process a single scene by name or path")
    parser.add_argument("--min-ncc", type=float, default=0.4, help="NCC correlation threshold for verification")
    parser.add_argument("--overwrite-ortho", action="store_true", help="Force re-generation of existing ortho TIFFs")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker processes")
    
    args = parser.parse_args()

    configure_pipeline_file_logger()
    dataset_dir = Path(args.dataset)
    output_root = Path(args.output_root)
    rpc_dir = Path(args.rpc_dir) if args.rpc_dir else (Path("data/reference_rpc") / args.matcher)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir missing: {dataset_dir}")
    if not rpc_dir.exists():
        raise FileNotFoundError(f"RPC dir missing: {rpc_dir}")

    scene_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    if args.scene:
        scene_path = Path(args.scene)
        if not scene_path.is_absolute():
            scene_path = dataset_dir / scene_path
        if not scene_path.exists() or not scene_path.is_dir():
            raise FileNotFoundError(f"Scene dir missing: {scene_path}")
        scene_dirs = [scene_path]
    
    print(f"\n[INFO] Starting Ortho-Validation Pipeline")
    print(f"[INFO] Scenes: {len(scene_dirs)} | Workers: {args.workers}")
    print(f"[INFO] Min NCC: {args.min_ncc} | Overwrite: {args.overwrite_ortho}\n", flush=True)

    summary_rows = []

    # 2. Multi-Process Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_scene = {
            executor.submit(
                worker_task, 
                sd, 
                rpc_dir, 
                output_root, 
                args.matcher, 
                args.min_ncc, 
                args.overwrite_ortho
            ): sd 
            for sd in scene_dirs
        }

        for future in concurrent.futures.as_completed(future_to_scene):
            summary_rows.append(future.result())

    # 3. Final Summary Reporting
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_root / f"verification_summary_{args.matcher}_{ts}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="") as f:
        fieldnames = ["scene", "status", "rmse", "ce90", "rmse_px", "ce90_px", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[INFO] Pipeline complete.")
    print(f"[INFO] Summary Report saved to: {report_path}")


if __name__ == "__main__":
    main()