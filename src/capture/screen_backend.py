"""Screen grab: dxcam first, then mss. All outputs RGB ``uint8`` HWC."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.capture.types import ScreenRegion

# dxcam often returns None sporadically; retry before treating as failure.
_DXCAM_GRAB_MAX_ATTEMPTS = 256


def create_dxcam_if_available() -> Any | None:
    try:
        import dxcam
    except ImportError:
        return None
    return dxcam.create()


def _crop_rgb(frame: np.ndarray, region: ScreenRegion) -> np.ndarray:
    return frame[
        region.top : region.top + region.height,
        region.left : region.left + region.width,
        :,
    ]


def grab_rgb_dxcam(camera: Any, region: ScreenRegion | None) -> np.ndarray | None:
    """Grab one frame via dxcam; convert BGR to RGB; optional crop."""

    import cv2

    frame = None
    for _ in range(_DXCAM_GRAB_MAX_ATTEMPTS):
        frame = camera.grab()
        if frame is not None:
            break
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if region is not None:
        rgb = _crop_rgb(rgb, region)
    return np.ascontiguousarray(rgb)


def grab_rgb_mss(sct: Any, region: ScreenRegion | None) -> np.ndarray:
    """Grab via mss; returns RGB uint8."""

    if region is not None:
        mon = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
    else:
        mon = sct.monitors[1]
    shot = sct.grab(mon)
    h, w = shot.height, shot.width
    rgb = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((h, w, 3))
    return np.ascontiguousarray(rgb)


def grab_screen_rgb(
    *,
    dxcam_camera: Any | None,
    mss_instance: Any | None,
    region: ScreenRegion | None,
) -> np.ndarray | None:
    """Try dxcam when ``dxcam_camera`` is set; else mss (requires ``mss_instance``)."""

    if dxcam_camera is not None:
        return grab_rgb_dxcam(dxcam_camera, region)
    if mss_instance is not None:
        return grab_rgb_mss(mss_instance, region)
    return None
