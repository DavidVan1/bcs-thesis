"""
PhiSat-2 Orthorectification Pipeline
=====================================

End-to-end pipeline for orthorectifying PhiSat-2 pushbroom satellite
imagery using Sentinel-2 tie points, DEM elevation, and independent
GCP verification.

Modules:
    config          – Scene definitions and project paths
    utils           – Image I/O, enhancement, tie-point loading
    sensor_model    – Pushbroom geometric model (forward + inverse)
    matchers        – Pluggable feature matchers (LightGlue, XoFTR, …)
    matching        – Feature matching pipeline with Sentinel-2
    calibration     – Robust 3-phase sensor calibration
    orthorectify    – DEM-aware orthorectification engine
    verify          – Independent GCP verification via cross-correlation
    run             – CLI entry point to run the full pipeline
"""

from .config import SceneConfig, get_scene_config, list_scenes


def get_matcher(*args, **kwargs):
    from .matchers import get_matcher as _get_matcher
    return _get_matcher(*args, **kwargs)


def list_matchers():
    from .matchers import list_matchers as _list_matchers
    return _list_matchers()

__all__ = [
    "SceneConfig",
    "get_scene_config",
    "list_scenes",
    "get_matcher",
    "list_matchers",
]
