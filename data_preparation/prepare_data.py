from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.fetch import (
    _load_acquisition_time,
    _load_footprint,
    download_dem,
    download_gcps,
    download_sentinel,
)


def fetch_if_missing(target: Path, fetch_fn, force: bool):
    exists = target.exists() and (target.is_file() or any(target.iterdir()))
    if exists and not force:
        print(f"  [SKIP] {target.name}")
        return
    fetch_fn()
    print(f"  [OK]   {target.name} fetched")


def prepare_scene(scene_dir: Path, force: bool = False):
    lon_min, lat_min, lon_max, lat_max = _load_footprint(scene_dir)
    acq_date = _load_acquisition_time(scene_dir)
    print(f"  Footprint: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")

    fetch_if_missing(
        scene_dir / "sentinel.tif",
        lambda: download_sentinel(lon_min, lat_min, lon_max, lat_max, acq_date, scene_dir / "sentinel.tif"),
        force,
    )
    fetch_if_missing(
        scene_dir / "dem.tif",
        lambda: download_dem(lon_min, lat_min, lon_max, lat_max, scene_dir / "dem.tif"),
        force,
    )
    fetch_if_missing(
        scene_dir / "sentinel_gri",
        lambda: download_gcps(lon_min, lat_min, lon_max, lat_max, scene_dir / "sentinel_gri"),
        force,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/scenes")
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional single scene folder name"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--exclude-csv",
        type=str,
        default=None,
        help="CSV file containing scene names to exclude",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    # Load exclusion list without pandas
    exclude_set = set()
    if args.exclude_csv:
        exclude_path = Path(args.exclude_csv)

        if not exclude_path.exists():
            raise FileNotFoundError(f"Exclude CSV not found: {exclude_path}")

        with exclude_path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()

                # Skip empty lines and header rows
                if not name or name.lower() == "scene":
                    continue

                # Handle comma-separated values by taking the first column
                name = name.split(",")[0].strip()
                exclude_set.add(name)

        print(f"Excluding {len(exclude_set)} scenes from {exclude_path}")

    # Gather scenes
    if args.scene:
        scenes = [dataset_dir / args.scene]
    else:
        scenes = sorted(d for d in dataset_dir.iterdir() if d.is_dir())

    # Filter excluded scenes
    if exclude_set:
        scenes = [s for s in scenes if s.name not in exclude_set]

    print(f"Found {len(scenes)} scenes\n")
    print(scenes)

    # Uncomment when ready to process scenes
    for scene_dir in scenes:
        print(f" {scene_dir.name}")
        try:
            prepare_scene(scene_dir, force=args.force)
        except Exception as e:
            print(f"  [FAIL] {e}")



if __name__ == "__main__":
    main()
