from pathlib import Path


# Project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_FOCAL_LENGTH: float = 105454.0   # pixels
DEFAULT_PRINCIPAL_POINT: float = 2048.0  # cx = cy = image centre
DEFAULT_MARGIN_PIXELS: int = 512
DEFAULT_MAX_KEYPOINTS: int = 2048
PHISAT_GSD_M: float = 4.75              # ground sample distance (metres)

