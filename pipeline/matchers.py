"""
Pluggable feature-matcher wrappers.

Each matcher takes two RGB uint8 images (H, W, 3) and returns
``{"keypoints0": (N,2), "keypoints1": (N,2), "confidence": (N,)}``.
Internally each class handles its own preprocessing (resize, grayscale
conversion, tensor format) and rescales output keypoints back to the
**original** image resolution so the caller never has to worry about it.

Supported matchers
------------------
* ``lightglue``   – SuperPoint + LightGlue  (sparse)
* ``xoftr``       – XoFTR                    (dense)
* ``loftr``       – MiniMA-LoFTR             (dense)
* ``roma``        – MiniMA-RoMa              (dense)
* ``mast3r``      – MASt3R                   (dense 3-D)
* ``dust3r``      – DUSt3R                   (dense 3-D)

Use :func:`get_matcher` as the single entry point::

    matcher = get_matcher("xoftr", device="cuda", max_keypoints=2000)
    result  = matcher.match(phisat_rgb, sentinel_rgb)
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# ── Paths into third_party/ ──────────────────────────────────────────────
_TP = Path(__file__).resolve().parent.parent / "third_party"
_HLOC = _TP / "hloc"


def _ensure_path(*dirs: Path) -> None:
    """Prepend directories to ``sys.path`` (idempotent)."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


# ── helpers ──────────────────────────────────────────────────────────────

def _to_grayscale_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """RGB uint8 (H,W,3) → float32 [1,1,H,W] on *device*."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, None].to(device)


def _to_rgb_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """RGB uint8 (H,W,3) → float32 [1,3,H,W] on *device*."""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _resize_divisible(img: np.ndarray, max_side: int, dfactor: int,
                      force_size: tuple[int, int] | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize *img* so it fits within *max_side* and dimensions are divisible
    by *dfactor*.  If *force_size* ``(W, H)`` is given, resize to exactly
    that before the dfactor pass.

    Returns ``(resized_img, scale)`` where ``scale`` maps from the
    *original* pixel coords to the *resized* pixel coords:
    ``kp_orig = kp_resized * scale``.
    """
    h, w = img.shape[:2]
    orig_size = np.array([w, h], dtype=np.float64)

    # 1. Optional max-side downscale
    s = max_side / max(w, h)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)

    # 2. Optional forced resize
    if force_size is not None:
        img = cv2.resize(img, force_size, interpolation=cv2.INTER_AREA)

    # 3. Ensure divisible by dfactor
    rh, rw = img.shape[:2]
    new_h = rh // dfactor * dfactor
    new_w = rw // dfactor * dfactor
    if (new_h, new_w) != (rh, rw):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    resized_size = np.array([img.shape[1], img.shape[0]], dtype=np.float64)
    scale = orig_size / resized_size  # multiply resized kp → original coords
    return img, scale


def _rescale_keypoints(kp: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale keypoints from resized coordinates back to original."""
    return kp * scale[np.newaxis, :]


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class BaseMatcher(ABC):
    """Interface every matcher must implement."""

    name: str = "base"

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        self.device = device
        self.max_keypoints = max_keypoints

    @abstractmethod
    def match(self, img0: np.ndarray,
              img1: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Match two RGB uint8 images.

        Returns
        -------
        dict with:
            keypoints0 : (N, 2) float  – x, y in *original* img0 coords
            keypoints1 : (N, 2) float  – x, y in *original* img1 coords
            confidence : (N,)   float
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# LightGlue  (sparse: SuperPoint + LightGlue)
# ═══════════════════════════════════════════════════════════════════════════

class LightGlueMatcher(BaseMatcher):
    name = "lightglue"

    def __init__(self, device: str = "cuda", max_keypoints: int = 2048):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP / "LightGlue")
        from lightglue import LightGlue, SuperPoint  # noqa
        print(f"  LightGlue on {device}")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def match(self, img0, img1):
        t0 = _to_grayscale_tensor(img0, self.device)
        t1 = _to_grayscale_tensor(img1, self.device)
        with torch.no_grad():
            f0 = self.extractor.extract(t0)
            f1 = self.extractor.extract(t1)
            m = self.matcher({"image0": f0, "image1": f1})
        matches = m["matches"][0]
        if len(matches) == 0:
            return _empty()
        kp0 = f0["keypoints"][0][matches[:, 0]].cpu().numpy()
        kp1 = f1["keypoints"][0][matches[:, 1]].cpu().numpy()
        conf = m["scores"][0].cpu().numpy()
        return {"keypoints0": kp0, "keypoints1": kp1, "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# XoFTR  (dense, grayscale)
# ═══════════════════════════════════════════════════════════════════════════

class XoFTRMatcher(BaseMatcher):
    """XoFTR — cross-modal optical-flow transformer (grayscale, 640×480)."""
    name = "xoftr"

    # preprocessing from hloc config
    _WIDTH, _HEIGHT = 640, 480
    _DFACTOR = 8

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP)
        from hloc.matchers.xoftr import XoFTR  # noqa
        conf = {
            "model_name": "weights_xoftr_640.ckpt",
            "match_threshold": 0.3,
            "max_keypoints": max_keypoints,
        }
        self.net = XoFTR(conf).eval().to(device)
        print(f"  XoFTR on {device}")

    def match(self, img0, img1):
        img0r, sc0 = _resize_divisible(img0, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        img1r, sc1 = _resize_divisible(img1, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        t0 = _to_grayscale_tensor(img0r, self.device)
        t1 = _to_grayscale_tensor(img1r, self.device)
        with torch.no_grad():
            pred = self.net({"image0": t0, "image1": t1})
        kp0 = pred["keypoints0"].cpu().numpy()
        kp1 = pred["keypoints1"].cpu().numpy()
        conf = pred.get("scores", torch.ones(len(kp0))).cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# LoFTR / MiniMA-LoFTR  (dense, grayscale)
# ═══════════════════════════════════════════════════════════════════════════

class LoFTRMatcher(BaseMatcher):
    """MiniMA-LoFTR — lightweight LoFTR variant (grayscale, 640×480)."""
    name = "loftr"

    _WIDTH, _HEIGHT = 640, 480
    _DFACTOR = 8

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP)
        from hloc.matchers.loftr import LoFTR  # noqa
        conf = {
            "weights": "outdoor",
            "model_name": "minima_loftr.ckpt",
            "max_keypoints": max_keypoints,
            "match_threshold": 0.2,
        }
        self.net = LoFTR(conf).eval().to(device)
        print(f"  MiniMA-LoFTR on {device}")

    def match(self, img0, img1):
        img0r, sc0 = _resize_divisible(img0, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        img1r, sc1 = _resize_divisible(img1, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        t0 = _to_grayscale_tensor(img0r, self.device)
        t1 = _to_grayscale_tensor(img1r, self.device)
        with torch.no_grad():
            pred = self.net({"image0": t0, "image1": t1})
        kp0 = pred["keypoints0"].cpu().numpy()
        kp1 = pred["keypoints1"].cpu().numpy()
        conf = pred.get("scores", torch.ones(len(kp0))).cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# RoMa / MiniMA-RoMa  (dense, RGB)
# ═══════════════════════════════════════════════════════════════════════════

class RoMAMatcher(BaseMatcher):
    """MiniMA-RoMa — lightweight RoMa variant (RGB, 320×240)."""
    name = "roma"

    _WIDTH, _HEIGHT = 320, 240
    _DFACTOR = 8

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP, _TP / "RoMa")
        from hloc.matchers.roma import Roma  # noqa
        conf = {
            "model_name": "minima_roma.pth",
            "model_utils_name": "dinov2_vitl14_pretrain.pth",
            "max_keypoints": max_keypoints,
            "coarse_res": (560, 560),
            "upsample_res": (864, 1152),
        }
        self.net = Roma(conf).eval().to(device)
        print(f"  MiniMA-RoMa on {device}")

    def match(self, img0, img1):
        # RoMa handles its own internal resizing, but we still
        # force-resize to 320x240 following the hloc config and
        # scale keypoints back afterwards.
        img0r, sc0 = _resize_divisible(img0, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        img1r, sc1 = _resize_divisible(img1, 1024, self._DFACTOR,
                                        force_size=(self._WIDTH, self._HEIGHT))
        t0 = _to_rgb_tensor(img0r, self.device)
        t1 = _to_rgb_tensor(img1r, self.device)
        with torch.no_grad():
            pred = self.net({"image0": t0, "image1": t1})
        kp0 = pred["keypoints0"].cpu().numpy()
        kp1 = pred["keypoints1"].cpu().numpy()
        conf = pred.get("mconf", torch.ones(len(kp0))).cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# MASt3R  (dense 3-D matching, RGB)
# ═══════════════════════════════════════════════════════════════════════════

class MASt3RMatcher(BaseMatcher):
    """MASt3R — multi-view stereo matcher (RGB, ≤512 px, dfactor 16)."""
    name = "mast3r"

    _MAX_SIDE = 512
    _DFACTOR = 16

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP, _TP / "mast3r", _TP / "dust3r")
        from hloc.matchers.mast3r import Mast3r  # noqa
        conf = {
            "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "max_keypoints": max_keypoints,
            "vit_patch_size": 16,
        }
        self.net = Mast3r(conf).eval().to(device)
        print(f"  MASt3R on {device}")

    def match(self, img0, img1):
        img0r, sc0 = _resize_divisible(img0, self._MAX_SIDE, self._DFACTOR)
        img1r, sc1 = _resize_divisible(img1, self._MAX_SIDE, self._DFACTOR)
        t0 = _to_rgb_tensor(img0r, self.device)
        t1 = _to_rgb_tensor(img1r, self.device)
        with torch.no_grad():
            pred = self.net({"image0": t0, "image1": t1})
        kp0 = pred["keypoints0"].cpu().numpy()
        kp1 = pred["keypoints1"].cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        conf = np.ones(len(kp0), dtype=np.float32)
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# DUSt3R  (dense 3-D matching, RGB)
# ═══════════════════════════════════════════════════════════════════════════

class DUSt3RMatcher(BaseMatcher):
    """DUSt3R — dense stereo matcher (RGB, ≤512 px, dfactor 16)."""
    name = "dust3r"

    _MAX_SIDE = 512
    _DFACTOR = 16

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP, _TP / "dust3r")
        from hloc.matchers.duster import Duster  # noqa
        conf = {
            "model_name": "duster_vit_large.pth",
            "max_keypoints": max_keypoints,
            "vit_patch_size": 16,
        }
        self.net = Duster(conf).eval().to(device)
        print(f"  DUSt3R on {device}")

    def match(self, img0, img1):
        img0r, sc0 = _resize_divisible(img0, self._MAX_SIDE, self._DFACTOR)
        img1r, sc1 = _resize_divisible(img1, self._MAX_SIDE, self._DFACTOR)
        t0 = _to_rgb_tensor(img0r, self.device)
        t1 = _to_rgb_tensor(img1r, self.device)
        with torch.no_grad():
            pred = self.net({"image0": t0, "image1": t1})
        kp0 = pred["keypoints0"].cpu().numpy()
        kp1 = pred["keypoints1"].cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        conf = np.ones(len(kp0), dtype=np.float32)
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

_REGISTRY: dict[str, type[BaseMatcher]] = {
    "lightglue": LightGlueMatcher,
    "xoftr":     XoFTRMatcher,
    "loftr":     LoFTRMatcher,
    "roma":      RoMAMatcher,
    "mast3r":    MASt3RMatcher,
    "dust3r":    DUSt3RMatcher,
}


def list_matchers() -> list[str]:
    """Return names of all registered matchers."""
    return list(_REGISTRY.keys())


def get_matcher(name: str, *,
                device: str = "cpu",
                max_keypoints: int = 2000) -> BaseMatcher:
    """
    Instantiate a matcher by name.

    Parameters
    ----------
    name : str
        One of: lightglue, xoftr, loftr, roma, mast3r, dust3r.
    device : str
        ``"cpu"`` or ``"cuda"``.
    max_keypoints : int
        Upper limit on returned matches.

    Returns
    -------
    BaseMatcher instance ready to call ``.match(img0, img1)``.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown matcher '{name}'. "
            f"Available: {', '.join(_REGISTRY)}"
        )
    return _REGISTRY[key](device=device, max_keypoints=max_keypoints)


# ── tiny helper ──────────────────────────────────────────────────────────

def _empty() -> Dict[str, np.ndarray]:
    return {"keypoints0": np.empty((0, 2)),
            "keypoints1": np.empty((0, 2)),
            "confidence": np.empty(0)}
