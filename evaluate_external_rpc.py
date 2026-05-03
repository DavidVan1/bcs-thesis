import argparse
import importlib.util
import time
import resource
import csv
import os
import datetime
import concurrent.futures
from pathlib import Path
from pipeline.evaluation.validation import run_validation
from pipeline.evaluation.orthorectify import run_orthorectify

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def load_external_script(path: Path):
    """Load the external algorithm module.

    Args:
        path: File path to the contributor's Python script.

    Returns:
        The imported module object.
    """
    spec = importlib.util.spec_from_file_location("external_algo", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def find_reference_ortho(reference_dir: Path, scene_id: str) -> Path | None:
    """Resolve a reference ortho TIFF path for a scene, if available.

    Args:
        reference_dir: Directory containing reference ortho files.
        scene_id: Scene identifier used to locate the ortho TIFF.

    Returns:
        The matched ortho path, or None if no candidate is found.
    """
    direct = reference_dir / f"reference_ortho.tif"
    if direct.exists():
        return direct
    scene_dir = reference_dir / scene_id
    if scene_dir.exists():
        matches = sorted(scene_dir.glob("ortho_reference_*.tif"))
        if matches:
            return matches[0]
    matches = sorted(reference_dir.glob(f"*{scene_id}*ortho*.tif"))
    return matches[0] if matches else None

def worker_task(scene_dir: Path, script_path: Path, out_root: Path,
                min_ncc: float, reference_ortho_dir: Path | None) -> dict:
    """Run external RPC generation and evaluation for one scene.

    Args:
        scene_dir: Input scene directory.
        script_path: Path to the external make_rpc_template.py script.
        out_root: Root output directory for generated files and metrics.
        min_ncc: Minimum NCC threshold passed to validation.
        reference_ortho_dir: Optional directory with reference ortho TIFFs.

    Returns:
        A dict containing timing, memory, validation, and comparison metrics.
    """
    scene_id = scene_dir.name
    rpc_path = out_root / f"rpc/{scene_id}_RPC.txt"
    ortho_path = out_root / f"ortho/{scene_id}.tif"
    val_path = out_root / f"metrics/{scene_id}_metrics.json"
    ref_val_path = out_root / f"metrics/{scene_id}_reference_metrics.json"

    print(f"[ START ] Scene: {scene_id}")

    try:
        algo = load_external_script(script_path)
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        wall_start = time.perf_counter()
        algo.process_scene(scene_dir, rpc_path)
        wall_end = time.perf_counter()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
        wall_seconds = wall_end - wall_start
        max_rss_kb = cpu_after.ru_maxrss
        max_rss_mb = max_rss_kb / 1024.0

        if not rpc_path.exists():
            raise FileNotFoundError(f"Script failed to create {rpc_path}")

        run_orthorectify(
            phisat_tiff=next((scene_dir / "bands").glob("*_RGB.tiff")),
            dem_path=scene_dir / "dem.tif",
            rpc_path=rpc_path,
            output_path=ortho_path
        )

        res = run_validation(
            ortho_path=ortho_path,
            gcp_json_path=next((scene_dir / "sentinel_gri").glob("*.json")),
            gcp_chip_dir=scene_dir / "sentinel_gri" / "L1C_chips",
            output_path=val_path,
            min_ncc=min_ncc,
        )

        stats = res.get("ncc", {}).get("stats", {}).get("total", {})
        rmse = stats.get("rmse")
        rmse_px = stats.get("rmse_px")
        ce90 = stats.get("ce90")
        ce90_px = stats.get("ce90_px")
        ref_rmse = None
        ref_rmse_px = None
        ref_ce90 = None
        ref_ce90_px = None
        if reference_ortho_dir is not None:
            ref_ortho = find_reference_ortho(reference_ortho_dir, scene_id)
            if ref_ortho is not None and ref_ortho.exists():
                ref_res = run_validation(
                    ortho_path=ref_ortho,
                    gcp_json_path=next((scene_dir / "sentinel_gri").glob("*.json")),
                    gcp_chip_dir=scene_dir / "sentinel_gri" / "L1C_chips",
                    output_path=ref_val_path,
                    min_ncc=min_ncc,
                )
                ref_stats = ref_res.get("ncc", {}).get("stats", {}).get("total", {})
                ref_rmse = ref_stats.get("rmse")
                ref_rmse_px = ref_stats.get("rmse_px")
                ref_ce90 = ref_stats.get("ce90")
                ref_ce90_px = ref_stats.get("ce90_px")

        print(f"[ OK ] {scene_id}")

        return {
            "scene_id": scene_id,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "max_rss_kb": max_rss_kb,
            "max_rss_mb": max_rss_mb,
            "rmse_m": rmse,
            "rmse_px": rmse_px,
            "ce90_m": ce90,
            "ce90_px": ce90_px,
            "ref_rmse_m": ref_rmse,
            "ref_rmse_px": ref_rmse_px,
            "ref_ce90_m": ref_ce90,
            "ref_ce90_px": ref_ce90_px,
            "delta_rmse_m": (rmse - ref_rmse) if (rmse is not None and ref_rmse is not None) else None,
            "delta_rmse_px": (rmse_px - ref_rmse_px) if (rmse_px is not None and ref_rmse_px is not None) else None,
            "delta_ce90_m": (ce90 - ref_ce90) if (ce90 is not None and ref_ce90 is not None) else None,
            "delta_ce90_px": (ce90_px - ref_ce90_px) if (ce90_px is not None and ref_ce90_px is not None) else None,
            "n_gcp_matches": stats.get("n"),
        }
    except Exception as e:
        print(f"[ FAIL ] {scene_id}: {e}")
        return {
            "scene_id": scene_id,
            "wall_seconds": None,
            "cpu_seconds": None,
            "max_rss_kb": None,
            "max_rss_mb": None,
            "rmse_m": None,
            "rmse_px": None,
            "ce90_m": None,
            "ce90_px": None,
            "ref_rmse_m": None,
            "ref_rmse_px": None,
            "ref_ce90_m": None,
            "ref_ce90_px": None,
            "delta_rmse_m": None,
            "delta_rmse_px": None,
            "delta_ce90_m": None,
            "delta_ce90_px": None,
            "n_gcp_matches": None,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to raw scenes")
    parser.add_argument("--script", required=True, help="Path to the external RPC generator script to evaluate")
    parser.add_argument("--output", required=True, help="Folder for results")
    parser.add_argument("--min-ncc", type=float, default=0.4, help="Minimum NCC threshold")
    parser.add_argument("--reference-ortho-dir", type=str, default=None, help="Optional dir with reference ortho TIFFs")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--scene", default=None, help="Optional scene name or absolute path")
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for sub in ["rpc", "ortho", "metrics"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = out_root / f"metrics_{ts}.csv"
    
    dataset_dir = Path(args.dataset)
    scenes = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if args.scene:
        scene_path = Path(args.scene)
        if not scene_path.is_absolute():
            scene_path = dataset_dir / scene_path
        if not scene_path.exists() or not scene_path.is_dir():
            raise FileNotFoundError(f"Scene dir missing: {scene_path}")
        scenes = [scene_path]
    reference_ortho_dir = Path(args.reference_ortho_dir) if args.reference_ortho_dir else None

    fieldnames = [
        "scene_id",
        "wall_seconds",
        "cpu_seconds",
        "max_rss_kb",
        "max_rss_mb",
        "rmse_m",
        "rmse_px",
        "ce90_m",
        "ce90_px",
        "ref_rmse_m",
        "ref_rmse_px",
        "ref_ce90_m",
        "ref_ce90_px",
        "delta_rmse_m",
        "delta_rmse_px",
        "delta_ce90_m",
        "delta_ce90_px",
        "n_gcp_matches",
    ]

    write_header = not metrics_csv.exists()

    with open(metrics_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_scene = {
                executor.submit(
                    worker_task,
                    scene_dir,
                    Path(args.script),
                    out_root,
                    args.min_ncc,
                    reference_ortho_dir,
                ): scene_dir
                for scene_dir in scenes
            }
            for future in concurrent.futures.as_completed(future_to_scene):
                row = future.result()
                writer.writerow(row)
                f.flush() 

if __name__ == "__main__":
    main()