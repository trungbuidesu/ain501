"""Tier 1 integration smoke tests (no heavy models)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.agent_manager import AgentManager


class _FastObject:
    def detect_instances(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        return []


def test_stress_manager_many_frames() -> None:
    mgr = AgentManager(
        object_agent=_FastObject(),
        action_agent=None,
        ocr_agent=None,
        face_pose_agent=None,
        timeout_object_s=0.5,
        max_workers=2,
    )
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    for _ in range(1000):
        out = mgr.run_parallel(frame_rgb=rgb, active=frozenset({"object"}))
        assert out["object"].status == "ok"
    mgr.close()


def test_router_selective_activation_mock() -> None:
    """Router output is represented as active agent set; manager respects it."""

    class _Ocr:
        def read_text(self, rgb: np.ndarray) -> list[dict[str, Any]]:
            return [
                {
                    "text": "x",
                    "position": {"x": 0, "y": 0, "w": 1, "h": 1},
                    "confidence": 0.9,
                }
            ]

    mgr = AgentManager(
        object_agent=_FastObject(),
        action_agent=None,
        ocr_agent=_Ocr(),
        face_pose_agent=None,
        timeout_object_s=1.0,
        timeout_ocr_s=1.0,
        max_workers=4,
    )
    rgb = np.ones((32, 32, 3), dtype=np.uint8) * 200
    out_only = mgr.run_parallel(frame_rgb=rgb, active=frozenset({"object"}))
    assert "ocr" not in out_only
    out_both = mgr.run_parallel(frame_rgb=rgb, active=frozenset({"object", "ocr"}))
    assert "ocr" in out_both and out_both["ocr"].status == "ok"
    mgr.close()
