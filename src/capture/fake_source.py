"""Synthetic :class:`FrameSource` for unit tests (no hardware)."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from src.capture.types import FramePacket, SourceKind


class FakeFrameSource:
    """Emits a fixed or alternating RGB pattern for a set number of frames."""

    def __init__(
        self,
        height: int,
        width: int,
        n_frames: int,
        *,
        source: SourceKind = "webcam",
        pattern: Literal["solid", "alternate"] = "solid",
        solid_value: int = 128,
    ) -> None:
        self._height = height
        self._width = width
        self._n_frames = n_frames
        self._source: SourceKind = source
        self._pattern = pattern
        self._solid_value = solid_value
        self._emitted = 0

    def read(self) -> FramePacket | None:
        if self._emitted >= self._n_frames:
            return None
        if self._pattern == "solid":
            data = np.full(
                (self._height, self._width, 3),
                self._solid_value,
                dtype=np.uint8,
            )
        else:
            base = 255 if (self._emitted % 2 == 0) else 0
            data = np.full((self._height, self._width, 3), base, dtype=np.uint8)
        t = time.monotonic()
        fid = self._emitted
        self._emitted += 1
        return FramePacket(
            data=data,
            t_mono=t,
            source=self._source,
            frame_id=fid,
        )

    def close(self) -> None:
        return
