"""
Scene configuration and project-wide paths.

Each scene is a dataclass holding all file paths needed by every
pipeline stage.  Add a new scene by appending to SCENES below.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


# ── Project root (one level above pipeline/) ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SceneConfig:
    """All paths and parameters for one PhiSat-2 acquisition."""

    name: str

    # PhiSat-2
    phisat_dir: str                # e.g. "phisat/phisat_sf"
    phisat_image: str              # relative to phisat_dir
    aocs_json: str = "AOCS.json"
    metadata_json: Optional[str] = None  # session_XXXX_metadata.json

    # Sentinel-2
    sentinel_dir: Optional[str] = None   # e.g. "sentinel/sentinel_sf"
    sentinel_band: str = "TCI"

    # Tie points
    tie_points_csv: Optional[str] = None  # e.g. "outputs/tie_points_lightglue_sf8.csv"

    # DEM
    dem_file: Optional[str] = None        # e.g. "DEM/sf3.tif"

    # Calibration
    calib_json: Optional[str] = None      # e.g. "outputs/phisat_model_robust_sf.json"

    # GCP verification
    gcp_json: Optional[str] = None        # e.g. "gcp/sf/N37W123.json"
    gcp_chip_dir: Optional[str] = None    # e.g. "gcp/sf/L1C_chips"

    # Outputs  (all stored under outputs/<name>/)
    ortho_tif: Optional[str] = None       # e.g. "outputs/sf/phisat_ortho.tif"
    verification_json: Optional[str] = None  # set by set_matcher()

    # Camera defaults
    initial_f: float = 105790.0
    cx: float = 2048.0
    cy: float = 2048.0

    # Matching
    margin_pixels: int = 512
    max_keypoints: int = 2048

    # ── Resolved absolute paths ──────────────────────────────────────

    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, rel: Optional[str]) -> Optional[Path]:
        """Resolve a relative path against project root, or return None."""
        if rel is None:
            return None
        return self.root / rel

    @property
    def phisat_dir_path(self) -> Path:
        return self.root / self.phisat_dir

    @property
    def phisat_image_path(self) -> Path:
        return self.phisat_dir_path / self.phisat_image

    @property
    def aocs_path(self) -> Path:
        return self.phisat_dir_path / self.aocs_json

    @property
    def metadata_path(self) -> Optional[Path]:
        if self.metadata_json is None:
            return None
        return self.phisat_dir_path / self.metadata_json

    @property
    def sentinel_dir_path(self) -> Optional[Path]:
        return self.resolve(self.sentinel_dir)

    @property
    def tie_points_path(self) -> Optional[Path]:
        return self.resolve(self.tie_points_csv)

    @property
    def dem_path(self) -> Optional[Path]:
        return self.resolve(self.dem_file)

    @property
    def calib_path(self) -> Optional[Path]:
        return self.resolve(self.calib_json)

    @property
    def gcp_json_path(self) -> Optional[Path]:
        return self.resolve(self.gcp_json)

    @property
    def gcp_chip_dir_path(self) -> Optional[Path]:
        return self.resolve(self.gcp_chip_dir)

    @property
    def ortho_path(self) -> Optional[Path]:
        return self.resolve(self.ortho_tif)

    # ── Per-scene output directory ───────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """All pipeline outputs go here: outputs/<scene_name>/"""
        d = self.root / "outputs" / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def match_viz_path(self) -> Path:
        return self.output_dir / "matches.png"

    @property
    def debug_phisat_path(self) -> Path:
        return self.output_dir / "debug_phisat.jpg"

    @property
    def debug_sentinel_path(self) -> Path:
        return self.output_dir / "debug_sentinel.jpg"

    @property
    def verification_json_path(self) -> Path:
        if self.verification_json:
            return self.root / self.verification_json
        return self.output_dir / "verification_results.json"

    def set_matcher(self, matcher_name: str) -> None:
        """
        Rewrite output paths so they are matcher-specific.

        Call this before running the pipeline so every stage
        reads/writes to e.g.  ``outputs/sf/tie_points_xoftr.csv``,
        ``outputs/sf/calibration_xoftr.json``, etc.
        """
        od = f"outputs/{self.name}"
        self.tie_points_csv      = f"{od}/tie_points_{matcher_name}.csv"
        self.calib_json          = f"{od}/calibration_{matcher_name}.json"
        self.ortho_tif           = f"{od}/ortho_{matcher_name}.tif"
        self.verification_json   = f"{od}/verification_{matcher_name}.json"

    def check_inputs(self, stage: str = "all") -> List[str]:
        """Return list of missing files for a given pipeline stage."""
        missing = []

        def _check(path: Optional[Path], label: str):
            if path is not None and not path.exists():
                missing.append(f"{label}: {path}")

        if stage in ("all", "matching", "calibration", "orthorectify"):
            _check(self.phisat_image_path, "PhiSat image")
            _check(self.aocs_path, "AOCS")

        if stage in ("all", "matching"):
            _check(self.sentinel_dir_path, "Sentinel dir")

        if stage in ("all", "calibration"):
            _check(self.tie_points_path, "Tie points CSV")

        if stage in ("all", "orthorectify"):
            _check(self.dem_path, "DEM")
            _check(self.calib_path, "Calibration JSON")

        if stage in ("all", "verify"):
            _check(self.ortho_path, "Ortho GeoTIFF")
            _check(self.gcp_json_path, "GCP JSON")
            _check(self.gcp_chip_dir_path, "GCP chip dir")

        return missing


# ═══════════════════════════════════════════════════════════════════════════
# Scene definitions
# ═══════════════════════════════════════════════════════════════════════════

SCENES = {
    # "sf": SceneConfig(
    #     name="sf",
    #     phisat_dir="phisat/phisat_sf",
    #     phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
    #     metadata_json="session_3817_metadata.json",
    #     sentinel_dir="sentinel/sentinel_sf",
    #     tie_points_csv="outputs/sf/tie_points.csv",
    #     dem_file="DEM/sf3.tif",
    #     calib_json="outputs/sf/calibration.json",
    #     gcp_json="gcp/sf/N37W123.json",
    #     gcp_chip_dir="gcp/sf/L1C_chips",
    #     ortho_tif="outputs/sf/ortho.tif",
    #     initial_f=105790.0,
    # ),

    "sf": SceneConfig(
        name="sf",
        phisat_dir="phisat/phisat_sf",
        phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
        metadata_json="session_3817_metadata.json",
        sentinel_dir="sentinel/sentinel_sf",
        dem_file="DEM/sf.tif",
        gcp_json="gcp/sf/N37W123.json",
        gcp_chip_dir="gcp/sf/L1C_chips",
        tie_points_csv="outputs/sf/tie_points.csv",
        calib_json="outputs/sf/calibration.json",
        ortho_tif="outputs/sf/ortho.tif",
        initial_f=105790.0,
    ),

    # "la": SceneConfig(
    #     name="la",
    #     phisat_dir="phisat/phisat_la",
    #     phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
    #     metadata_json="session_2532_metadata.json",
    #     sentinel_dir="sentinel/sentinel_la",
    #     tie_points_csv="outputs/la/tie_points.csv",
    #     dem_file="DEM/la3.tif",
    #     calib_json="outputs/la/calibration.json",
    #     gcp_json="gcp/la/N33W119.json",
    #     gcp_chip_dir="gcp/la/L1C_chips",
    #     ortho_tif="outputs/la/ortho.tif",
    #     initial_f=105790.0,
    # ),

    # "la": SceneConfig(
    #     name="la",
    #     phisat_dir="phisat/phisat_la",
    #     phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
    #     metadata_json="session_2532_metadata.json",
    #     sentinel_dir="sentinel/sentinel_la",
    #     dem_file="DEM/la.tif",
    #     gcp_json="gcp/la/N33W119.json",
    #     gcp_chip_dir="gcp/la/L1C_chips",
    #     tie_points_csv="outputs/la/tie_points.csv",
    #     calib_json="outputs/la/calibration.json",
    #     ortho_tif="outputs/la/ortho.tif",
    #     initial_f=105790.0,
    # ),

    "la": SceneConfig(
        name="la",
        phisat_dir="phisat/phisat_la",
        phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
        metadata_json="session_2532_metadata.json",
        sentinel_dir="sentinel/sentinel_la",
        dem_file="DEM/la.tif",
        gcp_json="gcp/la/N33W119.json",
        gcp_chip_dir="gcp/la/L1C_chips",
        tie_points_csv="outputs/la/tie_points.csv",
        calib_json="outputs/la/calibration.json",
        ortho_tif="outputs/la/ortho.tif",
        initial_f=105790.0,
    ),

    # "sicily": SceneConfig(
    #     name="sicily",
    #     phisat_dir="phisat/phisat_sicily",
    #     phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
    #     metadata_json="session_2587_metadata.json",
    #     sentinel_dir="sentinel/sentinel_sicily",
    #     dem_file="DEM/sicily.tif",
    #     gcp_json="gcp/sicily/N37E013.json",
    #     gcp_chip_dir="gcp/sicily/L1C_chips",
    #     tie_points_csv="outputs/sicily/tie_points.csv",
    #     calib_json="outputs/sicily/calibration.json",
    #     ortho_tif="outputs/sicily/ortho.tif",
    #     initial_f=105790.0,
    # ),

    "sicily": SceneConfig(
        name="sicily",
        phisat_dir="phisat/phisat_sicily",
        phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
        metadata_json="session_2587_metadata.json",
        sentinel_dir="sentinel/sentinel_sicily_before",
        dem_file="DEM/sicily.tif",
        gcp_json="gcp/sicily/N37E013.json",
        gcp_chip_dir="gcp/sicily/L1C_chips",
        tie_points_csv="outputs/sicily/tie_points.csv",
        calib_json="outputs/sicily/calibration.json",
        ortho_tif="outputs/sicily/ortho.tif",
        initial_f=105790.0,
    ),

    "valencia": SceneConfig(
        name="valencia",
        phisat_dir="phisat/phisat_valencia",
        phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
        metadata_json="session_2363_metadata.json",
        sentinel_dir="sentinel/sentinel_valencia",
        dem_file="DEM/valencia.tif",
        gcp_json="gcp/valencia/N41E002.json",
        gcp_chip_dir="gcp/valencia/L1C_chips",
        tie_points_csv="outputs/valencia/tie_points.csv",
        calib_json="outputs/valencia/calibration.json",
        ortho_tif="outputs/valencia/ortho.tif",
        initial_f=105790.0,
    ),

    "spain": SceneConfig(
        name="spain",
        phisat_dir="phisat/phisat_spain",
        phisat_image="bands/Bp_0_0_4096_9050_0_0_4096_9050_12_RGB.tiff",
        metadata_json="session_2599_metadata.json",
        sentinel_dir="sentinel/sentinel_spain",
        dem_file="DEM/spain.tif",
        gcp_json="gcp/spain/N42W002.json",
        gcp_chip_dir="gcp/spain/L1C_chips",
        tie_points_csv="outputs/spain/tie_points.csv",
        calib_json="outputs/spain/calibration.json",
        ortho_tif="outputs/spain/ortho.tif",
        initial_f=105790.0,
    ),

    "random": SceneConfig(
        name="random",
        phisat_dir="phisat/phisat_random",
        phisat_image="bands/Bp_0_0_4096_4096_0_0_4096_4096_12_RGB.tiff",
        metadata_json="session_2540_metadata.json",
        sentinel_dir="sentinel/sentinel_random",
        dem_file="DEM/random.tif",
        gcp_json="gcp/random/N44E007.json",
        gcp_chip_dir="gcp/random/L1C_chips",
        tie_points_csv="outputs/random/tie_points.csv",
        calib_json="outputs/random/calibration.json",
        ortho_tif="outputs/random/ortho.tif",
        initial_f=105790.0,
    ),



    # ── Add new scenes here ──────────────────────────────────────────
}


def get_scene_config(name: str) -> SceneConfig:
    """Look up a scene by short name (e.g. 'sf', 'la')."""
    if name not in SCENES:
        available = ", ".join(sorted(SCENES.keys()))
        raise KeyError(f"Unknown scene '{name}'. Available: {available}")
    return SCENES[name]


def list_scenes() -> List[str]:
    """Return sorted list of available scene names."""
    return sorted(SCENES.keys())
