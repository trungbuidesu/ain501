"""Face/pose agent tests."""

from __future__ import annotations

import pytest

pytest.importorskip("mediapipe")

import numpy as np

from src.agents.face_pose_agent import FacePoseAgent


def test_face_pose_analyze_runs() -> None:
    try:
        agent = FacePoseAgent(strict_artifacts=False)
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        out = agent.analyze(rgb)
    except (OSError, RuntimeError) as exc:
        pytest.skip(str(exc))
    assert "faces" in out and "poses" in out
    assert isinstance(out["faces"], list)
    assert isinstance(out["poses"], list)
