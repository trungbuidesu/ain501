"""Tests for OpenCV demo UI helpers (no live camera)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.types import PipelineStatus
from src.ui.dashboard import draw_dashboard
from src.ui.demo_window import save_screenshot
from src.ui.overlay_renderer import draw_detections


def test_overlay_renderer_draws_bbox() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = {
        "label": "person",
        "confidence": 0.9,
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
    }
    result = draw_detections(frame, [det])
    assert result.sum() > 0


def test_dashboard_draws_status() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    status = PipelineStatus(
        fps=2.0,
        pipeline_latency_ms=45.0,
        active_agents=["object"],
        scene_type="static",
        detection_count=1,
        caption_source="template",
        verbosity="medium",
        is_muted=False,
    )
    result = draw_dashboard(
        frame,
        status,
        caption_vi="Phía trước có 1 người",
        caption_subline="[TEMPLATE] 1 agents | 40ms",
    )
    assert result.sum() > 0


def test_screenshot_creates_files(tmp_path: Path) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    p = save_screenshot(frame, "test caption", output_dir=tmp_path)
    assert p.suffix == ".png"
    assert p.with_suffix(".txt").is_file()
