from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import json
import resource
import threading
import time
import logging
import multiprocessing
import concurrent.futures
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import torch

from pipeline.calibration import run_calibration
from pipeline.matching import run_matching
from pipeline.rpc_fit import fit_rpc
from pipeline.utils import configure_pipeline_file_logger, log_pipeline_stage

STAGE_CHOICES = ("all", "match", "calibrate", "rpc_fit")
EXECUTION_STAGES = ("match", "calibrate", "rpc_fit")


def get_process_rss_mb() -> float:
    """Retrieve the Resident Set Size (RSS) memory usage in MB."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0

def track_peak_rss(stop_event: threading.Event, peak_state: dict[str, float], interval_sec: float = 0.2) -> None:
    """Continuously monitor and record the peak RSS memory usage."""
    while not stop_event.is_set():
        rss_mb = get_process_rss_mb()
        if rss_mb > peak_state["peak_mb"]:
            peak_state["peak_mb"] = rss_mb
        stop_event.wait(interval_sec)


def find_tiff(scene_dir: Path) -> Path:
    """Locate the primary TIFF image for a given scene."""
    bands_dir = scene_dir / "bands"
    rgb_candidates = sorted(bands_dir.glob("*_RGB.tiff"))
    if len(rgb_candidates) == 1:
        return rgb_candidates[0]
    if len(rgb_candidates) > 1:
        raise RuntimeError(f"{scene_dir.name}: expected 1 RGB TIFF in bands/, found {len(rgb_candidates)}")

    exclude = {"dem.tif", "sentinel.tif"}
    tiffs = [f for f in scene_dir.glob("*.tif") if f.name not in exclude]
    tiffs.extend(scene_dir.glob("*.tiff"))
    if len(tiffs) != 1:
        raise RuntimeError(f"{scene_dir.name}: expected 1 PhiSat TIFF, found {len(tiffs)}")
    return tiffs[0]

def save_rpc_txt(rpc_dict: dict, path: Path):
    """Save the calculated RPC parameters to a text file."""
    with open(path, "w") as f:
        for key, value in rpc_dict.items():
            if isinstance(value, list):
                f.write(f"{key}: {' '.join(str(v) for v in value)}\n")
            else:
                f.write(f"{key}: {value}\n")

def safe_read_json(path: Path) -> dict:
    """Safely read and parse a JSON file, returning an empty dict on failure."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

def _count_tie_points(tie_points_path: Path) -> int | None:
    if not tie_points_path.exists():
        return None
    with tie_points_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)

def _read_rmse_m(calibration_path: Path) -> float | None:
    rmse = safe_read_json(calibration_path).get("stats", {}).get("rmse_m")
    return float(rmse) if isinstance(rmse, (int, float)) else None

def _read_ce90_m(scene_output_dir: Path, matcher: str) -> float | None:
    ce90 = safe_read_json(scene_output_dir / f"verification_ncc_{matcher}.json").get("ncc", {}).get("stats", {}).get("total", {}).get("ce90")
    return float(ce90) if isinstance(ce90, (int, float)) else None

def _read_verify_rmse_m(scene_output_dir: Path, matcher: str) -> float | None:
    rmse = safe_read_json(scene_output_dir / f"verification_ncc_{matcher}.json").get("ncc", {}).get("stats", {}).get("total", {}).get("rmse")
    return float(rmse) if isinstance(rmse, (int, float)) else None

def _read_matching_stats(scene_output_dir: Path, matcher: str) -> dict[str, int]:
    payload = safe_read_json(scene_output_dir / f"matching_stats_{matcher}.json")
    metrics: dict[str, int] = {}
    if isinstance(payload.get("raw_matches"), int):
        metrics["raw_matches"] = payload["raw_matches"]
    if isinstance(payload.get("inliers_after_ransac"), int):
        metrics["inliers_after_ransac"] = payload["inliers_after_ransac"]
    return metrics

def format_log(status: str, scene: str, **kwargs) -> str:
    """Format a clean, structured log line for standard output (e.g., nohup logs)."""
    parts = [f"[{status.upper():^6}] {scene}"]
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
    return "  ".join(parts)

def _parse_stages(stage_args: Sequence[str]) -> tuple[list[str], str]:
    """Parse and validate the requested pipeline stages."""
    raw_tokens: list[str] = [token.strip() for value in stage_args for token in value.split(",") if token.strip()]
    if not raw_tokens:
        raw_tokens = ["all"]

    invalid = [token for token in raw_tokens if token not in STAGE_CHOICES]
    if invalid:
        raise SystemExit(f"Invalid --stage value(s): {', '.join(sorted(set(invalid)))}")

    if "all" in raw_tokens:
        return list(EXECUTION_STAGES), "all"

    deduped = [stage for stage in EXECUTION_STAGES if stage in raw_tokens]
    return deduped, "+".join(deduped)


def run_pipeline(
    scene_dir: Path,
    matcher: str,
    output_dir: Path,
    stages: Sequence[str] | None = None,
) -> dict[str, float | int | str | None]:
    """Execute the selected stages of the georeferencing pipeline for a single scene."""
    tiff = find_tiff(scene_dir)
    aocs = scene_dir / "AOCS.json"
    meta = next(scene_dir.glob("session_*.json"), None)
    sentinel = scene_dir / "sentinel.tif"
    dem = scene_dir / "dem.tif"

    out = output_dir / scene_dir.name
    out.mkdir(parents=True, exist_ok=True)

    tie_points_path = out / f"tie_points_{matcher}.csv"
    calibration_path = out / f"calibration_{matcher}.json"
    rpc_path = out / f"rpc_{matcher}.json"

    metrics: dict[str, float | int | str | None] = {
        "matcher": matcher, "tie_points": None, "rmse_m": None, "verify_rmse_m": None, "ce90_m": None,
    }

    selected_stages = set(stages or EXECUTION_STAGES)

    if "match" in selected_stages:
        if not sentinel.exists():
            raise FileNotFoundError(f"{scene_dir.name}: missing sentinel.tif")
        tie_points = run_matching(tiff, sentinel, out, tie_points_path, matcher_name=matcher)
        metrics["tie_points"] = len(tie_points)

    if "calibrate" in selected_stages:
        run_calibration(aocs, meta, tie_points_path, dem, calibration_path)
        if metrics["tie_points"] is None:
            metrics["tie_points"] = _count_tie_points(tie_points_path)
        metrics["rmse_m"] = _read_rmse_m(calibration_path)

    if "rpc_fit" in selected_stages:
        fit_rpc(tiff, calibration_path, dem, rpc_path, aocs, meta)
        rpc_dict = json.loads(rpc_path.read_text())
        reference_rpc_dir = output_dir.parent / "reference_rpc" / matcher
        reference_rpc_dir.mkdir(parents=True, exist_ok=True)
        save_rpc_txt(rpc_dict, reference_rpc_dir / f"{scene_dir.name}_RPC.txt")
        metrics["verify_rmse_m"] = _read_verify_rmse_m(out, matcher)
        metrics["ce90_m"] = _read_ce90_m(out, matcher)

    if "rpc_fit" not in selected_stages:
        metrics["verify_rmse_m"] = _read_verify_rmse_m(out, matcher)
        metrics["ce90_m"] = _read_ce90_m(out, matcher)
        
    return metrics

def worker_task(scene_dir: Path, matcher: str, output_dir: Path, selected_stages: list[str], stage_label: str, gpu_queue: multiprocessing.Queue) -> dict:
    print(format_log("START", scene_dir.name, matcher=matcher), flush=True)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    gpu_id = gpu_queue.get() if gpu_queue else None
    
    logging.getLogger().setLevel(logging.ERROR)
    
    peak_state = {"peak_mb": get_process_rss_mb()}
    peak_stop_event = threading.Event()
    peak_thread = threading.Thread(target=track_peak_rss, args=(peak_stop_event, peak_state), daemon=True)
    peak_thread.start()

    status = "OK"
    error = ""
    pipeline_metrics = {}

    try:
        pipeline_metrics = run_pipeline(scene_dir, matcher, output_dir, stages=selected_stages)
        print(format_log("OK", scene_dir.name, matcher=matcher, 
                         tie_points=pipeline_metrics.get("tie_points"), 
                         rmse_m=pipeline_metrics.get("rmse_m"), 
                         ce90_m=pipeline_metrics.get("ce90_m")), flush=True)
        
        stage_metrics = {
            "calib_rmse": pipeline_metrics.get("rmse_m"), 
            "verify_rmse": pipeline_metrics.get("verify_rmse_m"), 
            "ce90": pipeline_metrics.get("ce90_m"),
        }
        if pipeline_metrics.get("tie_points") is not None:
            stage_metrics["inliers_after_ransac"] = pipeline_metrics.get("tie_points")
        
        log_pipeline_stage(scene=scene_dir.name, matcher=matcher, stage=stage_label, status="OK", metrics=stage_metrics)
    except Exception as exc:
        status = "FAIL"
        error = str(exc)
        print(format_log("FAIL", scene_dir.name, matcher=matcher, error=error), flush=True)
        log_pipeline_stage(scene=scene_dir.name, matcher=matcher, stage=stage_label, status="FAIL", reason=error)
        
    finally:
        if gpu_queue is not None and gpu_id is not None:
            gpu_queue.put(gpu_id)
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        peak_stop_event.set()
        peak_thread.join(timeout=1.0)
        
        ram_peak_mb = max(peak_state["peak_mb"], get_process_rss_mb())
        gpu_peak_mb = 0.0
        
        if torch.cuda.is_available():
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()

        elapsed_sec = wall_end - wall_start
        cpu_time_sec = cpu_end - cpu_start
        cpu_percent = (cpu_time_sec / elapsed_sec * 100.0) if elapsed_sec > 0 else 0.0

    csv_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": scene_dir.name, "matcher": matcher, "stage": stage_label,
        "status": status, "error": error,
        "elapsed_sec": f"{elapsed_sec:.3f}", "cpu_time_sec": f"{cpu_time_sec:.3f}",
        "cpu_percent": f"{cpu_percent:.2f}", "ram_peak_mb": f"{ram_peak_mb:.2f}", "gpu_peak_mb": f"{gpu_peak_mb:.2f}",
    }

    return {
        "scene_name": scene_dir.name,
        "status": status,
        "error": error,
        "metrics": pipeline_metrics,
        "csv_row": csv_row
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the reference dataset for the georeferencing pipeline.")
    parser.add_argument("--dataset", required=False, help="Directory containing the input scene folders")
    parser.add_argument("--output", required=True, help="Root directory for the output data")
    parser.add_argument("--matcher", default="efficientloftr", help="Name of the feature matching algorithm to use")
    parser.add_argument("--stage", nargs="+", default=["all"], help="Pipeline stages to execute")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--scene", default=None, help="Optional: process a single scene by name or path")
    args = parser.parse_args()

    # Validate dataset/scene arguments
    if not args.scene and not args.dataset:
        raise SystemExit("Either --dataset or --scene must be provided")
    if args.scene and Path(args.scene).is_absolute() and not args.dataset:
        # Scene is absolute path, dataset not needed
        pass
    elif not args.dataset:
        raise SystemExit("--dataset is required when --scene is a relative path")

    selected_stages, stage_label = _parse_stages(args.stage)

    needs_gpu = "match" in selected_stages or "all" in selected_stages
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if needs_gpu:
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()

        gpu_ids = [0]
        
        for gpu_id in gpu_ids:
            for _ in range(args.workers): 
                gpu_queue.put(gpu_id)
        
        max_workers = gpu_queue.qsize()
        print(f"[INFO] Launching {max_workers} workers (on GPU(s)): {gpu_ids}")
    else:
        gpu_queue = None
        max_workers = args.workers


    print(f"[INFO] Using {max_workers} workers based on VRAM constraints.")
    

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.matcher}_{stage_label}_{run_timestamp}"
    configure_pipeline_file_logger(run_id=run_id)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene and Path(args.scene).is_absolute():
        # Single scene with absolute path
        scene_path = Path(args.scene)
        if not scene_path.exists() or not scene_path.is_dir():
            raise SystemExit(f"Scene not found: {scene_path}")
        scene_dirs = [scene_path]
    else:
        # Multiple scenes from dataset or single scene by relative name
        dataset_dir = Path(args.dataset)
        scene_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        
        if args.scene:
            scene_path = dataset_dir / args.scene
            if not scene_path.exists() or not scene_path.is_dir():
                raise SystemExit(f"Scene not found: {scene_path}")
            scene_dirs = [scene_path]

    failures = []
    metrics_rows = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_scene = {
            executor.submit(worker_task, sd, args.matcher, output_dir, selected_stages, stage_label, gpu_queue): sd 
            for sd in scene_dirs
        }

        for future in concurrent.futures.as_completed(future_to_scene):
            res = future.result()
            if res["status"] == "FAIL":
                failures.append(res["scene_name"])

            metrics_rows.append(res["csv_row"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv_path = output_dir / f"{stage_label}/scene_metrics_{args.matcher}_{ts}.csv"
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_csv_path.open("w", newline="") as f:
        fieldnames = [
            "timestamp_utc", "scene", "matcher", "stage", "status", "error",
            "elapsed_sec", "cpu_time_sec", "cpu_percent", "ram_peak_mb", "gpu_peak_mb"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\n[INFO] Execution complete. Comprehensive metrics saved to: {metrics_csv_path}", flush=True)

    if failures:
        print(f"[ERROR] The process completed with failures on {len(failures)} scenes.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()