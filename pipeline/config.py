"""
Scene configuration and project-wide paths.

Each scene is a dataclass holding all file paths needed by every
pipeline stage.  Add a new scene by editing pipeline/scenes.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


# ── Project root (one level above pipeline/) ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Camera / sensor defaults ────────────────────────────────────────────
DEFAULT_FOCAL_LENGTH: float = 105454.0   # pixels
DEFAULT_PRINCIPAL_POINT: float = 2048.0  # cx = cy = image centre
DEFAULT_MARGIN_PIXELS: int = 512
DEFAULT_MAX_KEYPOINTS: int = 2048
PHISAT_GSD_M: float = 4.75              # ground sample distance (metres)

