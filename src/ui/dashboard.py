"""Status bar and caption panel (BGR frames)."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import PipelineStatus
from src.ui.text_vi import overlay_alpha_rect, put_vietnamese_lines


def draw_dashboard(
    frame_bgr: np.ndarray,
    status: PipelineStatus,
    *,
    caption_vi: str,
    caption_subline: str,
) -> np.ndarray:
    """Add bottom caption area and status bar."""

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    verb = status.verbosity.upper()[:3]
    cv2.putText(
        out,
        f"[{verb}]",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if status.is_muted:
        cv2.putText(
            out,
            "MUTED",
            (w - 120, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    bar_h = 28
    overlay_alpha_rect(out, 0, h - bar_h, w - 1, h - 1, (0, 0, 0), alpha=0.45)
    agents = " ".join(sorted(status.active_agents))[:40]
    line = (
        f"FPS: {status.fps:.1f} | Pipeline: {status.pipeline_latency_ms:.0f}ms | "
        f"Agents: {agents} | Scene: {status.scene_type} | Det: {status.detection_count}"
    )
    cv2.putText(
        out,
        line[: min(120, len(line))],
        (6, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    panel_h = 52
    y0 = h - bar_h - panel_h - 6
    overlay_alpha_rect(
        out,
        0,
        max(0, y0),
        w - 1,
        h - bar_h - 2,
        (20, 20, 20),
        alpha=0.5,
    )
    lines: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []
    if caption_vi:
        lines.append((caption_vi, (8, y0 + 22), 18, (255, 255, 255)))
    if caption_subline:
        lines.append((caption_subline, (8, y0 + 44), 14, (200, 200, 200)))
    if lines:
        out = put_vietnamese_lines(out, lines)
    return out
