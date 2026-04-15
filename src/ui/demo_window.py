"""OpenCV demo window; keyboard controls; does not own inference."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.app.runtime import FrameProcessResult
from src.core.types import PipelineStatus
from src.ui.dashboard import draw_dashboard
from src.ui.overlay_renderer import compose_visual_overlay
from src.ui.state import DemoControlState


class DemoWindow:
    """High-level viewer using cv2.imshow."""

    window_name = "Accessibility Vision Assistant"

    def __init__(
        self,
        *,
        control: DemoControlState | None = None,
        screenshot_dir: Path | str = "screenshots",
    ) -> None:
        self.running = True
        self._control = control or DemoControlState()
        self._control.screenshot_dir = Path(screenshot_dir)
        self._cv2 = cv2
        self._gui_ok = False
        try:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            self._cv2.resizeWindow(self.window_name, 960, 720)
            self._gui_ok = True
        except cv2.error:
            print(
                "[demo] OpenCV GUI unavailable (e.g. headless build); "
                "continuing without a display window.",
                flush=True,
            )

    @property
    def gui_available(self) -> bool:
        return self._gui_ok

    def show_frame(
        self,
        frame_rgb: np.ndarray,
        detections: list[dict[str, Any]],
        frame_result: FrameProcessResult,
        caption_vi: str,
        status: PipelineStatus,
    ) -> None:
        """Overlay + dashboard + imshow + keyboard handling."""

        bgr = compose_visual_overlay(frame_rgb, detections, frame_result)
        sub = (
            f"[{status.caption_source.upper()}] "
            f"{len(status.active_agents)} agents | {status.pipeline_latency_ms:.0f}ms"
        )
        st = replace(status, is_muted=True) if self._control.muted else status
        display = draw_dashboard(
            bgr,
            st,
            caption_vi=caption_vi,
            caption_subline=sub,
        )
        if not self._gui_ok:
            return
        self._cv2.imshow(self.window_name, display)
        key = self._cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.running = False
        elif key == ord("m"):
            self._control.muted = not self._control.muted
        elif key == ord("v"):
            cycle = ("low", "medium", "high")
            try:
                i = (cycle.index(self._control.verbosity) + 1) % 3
            except ValueError:
                i = 0
            self._control.verbosity = cycle[i]
        elif key == ord("s"):
            save_screenshot(
                display,
                caption_vi,
                output_dir=self._control.screenshot_dir,
            )
        elif key == ord("d"):
            self._control.force_describe = True
        elif key == ord("r"):
            self._control.rag_enabled = not self._control.rag_enabled

    def close(self) -> None:
        if self._gui_ok:
            self._cv2.destroyAllWindows()


def save_screenshot(
    frame_bgr: np.ndarray,
    caption: str,
    *,
    output_dir: Path | str = "screenshots",
) -> Path:
    """Write PNG and sidecar UTF-8 caption text."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = out / f"screenshot_{ts}.png"
    txt = out / f"screenshot_{ts}.txt"
    cv2.imwrite(str(png), frame_bgr)
    txt.write_text(caption, encoding="utf-8")
    return png
