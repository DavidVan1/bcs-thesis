"""
Resource profiler for pipeline stages.

Tracks wall-clock time, peak RSS memory, CPU utilisation,
and (optionally) GPU memory / utilisation via nvidia-smi.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


# ── GPU helpers ─────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    """Return True if nvidia-smi is reachable."""
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _query_gpu() -> Optional[Dict[str, float]]:
    """
    Query GPU 0 utilisation and memory via nvidia-smi.

    Returns dict with keys:
        gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb
    or None when nvidia-smi is absent.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mb": float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
        }
    except Exception:
        return None


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class StageProfile:
    """Resource snapshot for one pipeline stage."""
    stage: str
    wall_time_s: float = 0.0
    peak_rss_mb: float = 0.0
    cpu_percent: float = 0.0          # average over the stage
    gpu_util_pct: Optional[float] = None
    gpu_peak_mem_mb: Optional[float] = None
    gpu_total_mem_mb: Optional[float] = None


@dataclass
class PipelineProfile:
    """Aggregated profiles for the whole pipeline run."""
    scene: str = ""
    matcher: str = ""
    stages: List[StageProfile] = field(default_factory=list)
    total_wall_time_s: float = 0.0

    def summary_table(self) -> str:
        """Return a human-readable summary table."""
        lines: List[str] = []

        header = (
            f"{'Stage':<16} {'Time (s)':>10} {'Peak RSS (MB)':>14} "
            f"{'CPU (%)':>8}"
        )
        has_gpu = any(s.gpu_util_pct is not None for s in self.stages)
        if has_gpu:
            header += f" {'GPU (%)':>8} {'GPU Mem (MB)':>13}"
        lines.append(header)
        lines.append("─" * len(header))

        for s in self.stages:
            row = (
                f"{s.stage:<16} {s.wall_time_s:>10.2f} "
                f"{s.peak_rss_mb:>14.1f} {s.cpu_percent:>8.1f}"
            )
            if has_gpu:
                gpu_u = f"{s.gpu_util_pct:.0f}" if s.gpu_util_pct is not None else "—"
                gpu_m = f"{s.gpu_peak_mem_mb:.0f}" if s.gpu_peak_mem_mb is not None else "—"
                row += f" {gpu_u:>8} {gpu_m:>13}"
            lines.append(row)

        lines.append("─" * len(lines[1]))
        lines.append(f"{'TOTAL':<16} {self.total_wall_time_s:>10.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "matcher": self.matcher,
            "total_wall_time_s": self.total_wall_time_s,
            "stages": [asdict(s) for s in self.stages],
        }

    def save(self, path: Path) -> None:
        """Write the profile to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Resource profile saved → %s", path)


# ── Context manager for stage profiling ─────────────────────────────────

@contextmanager
def profile_stage(name: str, *, use_gpu: bool = False):
    """
    Context manager that yields a StageProfile populated on exit.

    Usage::

        with profile_stage("match", use_gpu=True) as prof:
            do_matching()
        print(prof.wall_time_s)
    """
    proc = psutil.Process(os.getpid())
    prof = StageProfile(stage=name)

    # snapshot before
    proc.cpu_percent(interval=None)      # prime the counter
    rss_before = proc.memory_info().rss
    gpu_peak_mem: float = 0.0
    if use_gpu:
        snap = _query_gpu()
        if snap:
            gpu_peak_mem = snap["gpu_mem_used_mb"]

    t0 = time.perf_counter()
    try:
        yield prof
    finally:
        t1 = time.perf_counter()

        prof.wall_time_s = t1 - t0
        prof.peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
        prof.cpu_percent = proc.cpu_percent(interval=None)

        if use_gpu:
            snap = _query_gpu()
            if snap:
                mem_now = snap["gpu_mem_used_mb"]
                prof.gpu_peak_mem_mb = max(gpu_peak_mem, mem_now)
                prof.gpu_util_pct = snap["gpu_util_pct"]
                prof.gpu_total_mem_mb = snap["gpu_mem_total_mb"]
