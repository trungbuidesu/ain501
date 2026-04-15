"""Unified frame contract for webcam, screen, and file capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SourceKind = Literal["webcam", "screen", "file"]


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    """Screen capture crop in pixel coordinates (origin top-left of virtual screen)."""

    left: int
    top: int
    width: int
    height: int


@dataclass(slots=True)
class FramePacket:
    """Single RGB frame from any :class:`FrameSource`."""

    data: np.ndarray
    """uint8, shape ``(H, W, 3)``, RGB channel order."""

    t_mono: float
    """Monotonic timestamp from :func:`time.monotonic`."""

    source: SourceKind
    frame_id: int
    path: str | None = None
    """Set when ``source == \"file\"``."""

    region: ScreenRegion | None = None
    """Set when ``source == \"screen\"`` and a crop was applied."""

    def __post_init__(self) -> None:
        if self.data.dtype != np.uint8:
            raise TypeError("FramePacket.data must be uint8")
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError("FramePacket.data must be HWC with 3 channels")
