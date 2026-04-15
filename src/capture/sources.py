"""Webcam, file, and screen :class:`FrameSource` implementations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.capture.config import CapturePipelineConfig
from src.capture.screen_backend import (
    create_dxcam_if_available,
    grab_screen_rgb,
)
from src.capture.types import FramePacket, ScreenRegion, SourceKind


class WebcamSource:
    """``cv2.VideoCapture(index)``; frames as RGB."""

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        self._frame_id = 0
        self._index = index

    def read(self) -> FramePacket | None:
        ok, bgr = self._cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pkt = FramePacket(
            data=np.ascontiguousarray(rgb),
            t_mono=time.monotonic(),
            source="webcam",
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        return pkt

    def close(self) -> None:
        self._cap.release()


class FileVideoSource:
    """``cv2.VideoCapture(path)`` with optional loop at EOF."""

    def __init__(self, path: Path | str, *, loop: bool = False) -> None:
        self._path = Path(path)
        self._loop = loop
        self._cap = cv2.VideoCapture(str(self._path))
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        ok, bgr = self._cap.read()
        if not ok and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pkt = FramePacket(
            data=np.ascontiguousarray(rgb),
            t_mono=time.monotonic(),
            source="file",
            frame_id=self._frame_id,
            path=str(self._path.resolve()),
        )
        self._frame_id += 1
        return pkt

    def close(self) -> None:
        self._cap.release()


class ScreenSource:
    """Screen capture via dxcam or mss; optional ``ScreenRegion`` crop."""

    def __init__(
        self,
        region: ScreenRegion | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        self._region = region
        self._frame_id = 0
        self._dxcam: Any | None = None
        self._mss: Any | None = None
        backend_norm = backend.strip().lower()
        if backend_norm not in {"auto", "dxcam", "mss"}:
            raise ValueError(f"unknown screen backend: {backend}")
        # User region uses global virtual-desktop coords. dxcam's buffer is not
        # guaranteed to match that space; cropping dxcam output was wrong and could
        # hang or return None. Use mss for any explicit region (multi-monitor safe).
        if region is not None and backend_norm == "dxcam":
            raise ValueError("backend=dxcam is not supported with explicit region crop")

        if backend_norm == "mss":
            import mss

            self._mss = mss.mss()
        elif backend_norm == "dxcam":
            self._dxcam = create_dxcam_if_available()
            if self._dxcam is None:
                raise RuntimeError("dxcam backend requested but dxcam is not available")
        elif region is not None:
            import mss

            self._mss = mss.mss()
        else:
            self._dxcam = create_dxcam_if_available()
            if self._dxcam is None:
                import mss

                self._mss = mss.mss()

    def read(self) -> FramePacket | None:
        rgb = grab_screen_rgb(
            dxcam_camera=self._dxcam,
            mss_instance=self._mss,
            region=self._region,
        )
        if rgb is None:
            return None
        pkt = FramePacket(
            data=rgb,
            t_mono=time.monotonic(),
            source="screen",
            frame_id=self._frame_id,
            region=self._region,
        )
        self._frame_id += 1
        return pkt

    def close(self) -> None:
        self._dxcam = None
        if self._mss is not None:
            closer = getattr(self._mss, "close", None)
            if callable(closer):
                closer()
        self._mss = None


def build_source(
    kind: SourceKind,
    *,
    webcam_index: int = 0,
    file_path: Path | None = None,
    file_loop: bool = False,
    screen_region: ScreenRegion | None = None,
) -> WebcamSource | FileVideoSource | ScreenSource:
    """Factory for configured sources."""

    if kind == "webcam":
        return WebcamSource(webcam_index)
    if kind == "file":
        if file_path is None:
            raise ValueError("file_path required for file source")
        return FileVideoSource(file_path, loop=file_loop)
    if kind == "screen":
        return ScreenSource(screen_region)
    raise ValueError(f"unknown source kind: {kind}")


def frame_source_from_config(
    cfg: CapturePipelineConfig,
) -> WebcamSource | FileVideoSource | ScreenSource:
    """Build a source from a loaded :class:`CapturePipelineConfig`."""

    s = cfg.source
    if s.type == "webcam":
        return WebcamSource(s.webcam_index)
    if s.type == "file":
        if s.file_path is None:
            raise ValueError("source.file_path is required when source.type is file")
        return FileVideoSource(s.file_path, loop=s.file_loop)
    if s.type == "screen":
        return ScreenSource(cfg.screen.region, backend=cfg.screen.backend)
    raise ValueError(f"unknown source type: {s.type}")
