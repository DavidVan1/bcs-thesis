from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import cv2
import logging
import numpy as np
import torch

# ── Paths into third_party/ ────────────────────────────────────────────
_TP = Path(__file__).resolve().parent.parent / "third_party"

logger = logging.getLogger(__name__)


def _ensure_path(*dirs: Path) -> None:
    """Prepend directories to ``sys.path`` (idempotent)."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)



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


def _get_proportional_max_sides(img0: np.ndarray, img1: np.ndarray, base_max: int) -> tuple[int, int]:
    """Calculate max_side for both images so they maintain the same physical scale."""
    max_d0 = max(img0.shape[:2])
    max_d1 = max(img1.shape[:2])
    scale = base_max / max(1, max_d0, max_d1)
    if scale > 1.0:
        scale = 1.0
    return max(32, int(round(max_d0 * scale))), max(32, int(round(max_d1 * scale)))


def _empty() -> Dict[str, np.ndarray]:
    """Return an empty match result."""
    return {
        "keypoints0": np.empty((0, 2)),
        "keypoints1": np.empty((0, 2)),
        "confidence": np.empty(0),
    }


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


class LightGlueMatcher(BaseMatcher):
    name = "lightglue"

    def __init__(self, device: str = "cuda", max_keypoints: int = 2048):
        super().__init__(device, max_keypoints)
        _ensure_path(_TP / "LightGlue")
        from lightglue import LightGlue, SuperPoint  # noqa
        logger.info("  LightGlue on %s", device)
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


class EfficientLoFTRMatcher(BaseMatcher):
    """EfficientLoFTR — fast LoFTR variant with RepVGG backbone (grayscale, 832×832)."""
    name = "efficientloftr"

    _WIDTH, _HEIGHT = 832, 832
    _DFACTOR = 32  # backbone stride (8) × aggregation size (4)

    def __init__(self, device: str = "cuda", max_keypoints: int = 2000):
        super().__init__(device, max_keypoints)
        _eloftr_root = _TP / "EfficientLoFTR"
        _ensure_path(str(_eloftr_root))

        from src.config.default import get_cfg_defaults # noqa
        from src.utils.misc import lower_config  # noqa
        from src.loftr import LoFTR as ELoFTR  # noqa
        from src.loftr.loftr import reparameter  # noqa

        # Build config (full model, outdoor weights)
        config = get_cfg_defaults()
        cfg_path = _eloftr_root / "configs" / "loftr" / "eloftr_full.py"
        config.merge_from_file(str(cfg_path))
        config.LOFTR.COARSE.NPE = [832, 832, self._WIDTH, self._HEIGHT]
        _config = lower_config(config)

        # Instantiate and load weights
        ckpt_path = _eloftr_root / "weights" / "eloftr_outdoor.ckpt"
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except (TypeError, pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning(
                "  EfficientLoFTR checkpoint is not weights-only compatible (%s). "
                "Falling back to weights_only=False for trusted local checkpoint.",
                exc.__class__.__name__,
            )
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.net = ELoFTR(config=_config["loftr"])
        self.net.load_state_dict(state_dict, strict=False)
        self.net = reparameter(self.net)
        self.net.eval().to(device)
        logger.info("  EfficientLoFTR on %s", device)

    def match(self, img0, img1):
        side0, side1 = _get_proportional_max_sides(img0, img1, 1024)
        img0r, sc0 = _resize_divisible(img0, side0, self._DFACTOR
                                        )
        img1r, sc1 = _resize_divisible(img1, side1, self._DFACTOR
                                        )
        t0 = _to_grayscale_tensor(img0r, self.device)
        t1 = _to_grayscale_tensor(img1r, self.device)
        data = {"image0": t0, "image1": t1}
        with torch.no_grad():
            self.net(data)
        kp0 = data["mkpts0_f"].cpu().numpy()
        kp1 = data["mkpts1_f"].cpu().numpy()
        conf = data.get("mconf", torch.ones(len(kp0))).cpu().numpy()
        if len(kp0) == 0:
            return _empty()
        # Top-k by confidence
        if len(kp0) > self.max_keypoints:
            idx = np.argsort(conf)[::-1][: self.max_keypoints]
            kp0, kp1, conf = kp0[idx], kp1[idx], conf[idx]
        return {"keypoints0": _rescale_keypoints(kp0, sc0),
                "keypoints1": _rescale_keypoints(kp1, sc1),
                "confidence": conf}


_REGISTRY: dict[str, type[BaseMatcher]] = {
    "lightglue":      LightGlueMatcher,
    "efficientloftr": EfficientLoFTRMatcher,
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
        One of: lightglue, efficientloftr.
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
