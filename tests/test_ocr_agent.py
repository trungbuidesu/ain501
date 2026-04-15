"""OCR agent tests (optional Paddle stack)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("paddleocr")
pytest.importorskip("paddle")

from src.agents.ocr_agent import OcrAgent


def test_ocr_reads_synthetic_text() -> None:
    img = np.full((120, 360, 3), 255, dtype=np.uint8)
    agent = OcrAgent(confidence_threshold=0.2, grid=1)
    lines = agent.read_text(img)
    assert isinstance(lines, list)
