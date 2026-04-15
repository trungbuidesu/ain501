"""Compare consecutive RGB frames; skip pipeline work when nothing changed."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from src.capture.config import ChangeDetectionConfig


def _resize_max_side(rgb: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


class ChangeDetector:
    """Emit on first frame; skip when consecutive frames are identical."""

    def __init__(self, cfg: ChangeDetectionConfig) -> None:
        self._method = cfg.method
        self._ssim_below = cfg.ssim_trigger_below
        self._frac_above = cfg.changed_fraction_trigger_above
        self._pix_thr = cfg.pixel_diff_threshold
        self._max_side = cfg.max_metric_side

    def should_process(self, prev: np.ndarray | None, curr: np.ndarray) -> bool:
        if prev is None:
            return True
        a = _resize_max_side(prev, self._max_side)
        b = _resize_max_side(curr, self._max_side)
        if a.shape != b.shape:
            return True
        if self._method == "ssim":
            g1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            s = float(
                structural_similarity(
                    g1,
                    g2,
                    data_range=255,
                )
            )
            return s < self._ssim_below
        g1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16))
        frac = float((diff > self._pix_thr).mean())
        return frac > self._frac_above
