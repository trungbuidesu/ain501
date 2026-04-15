"""Tests for parallel agent manager."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agents.agent_manager import AgentManager
from src.capture.types import FramePacket


class _SlowObject:
    def detect_instances(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        time.sleep(0.08)
        return [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                "count": 1,
            }
        ]


class _SlowAction:
    def predict(self, packets: list[FramePacket]) -> dict[str, Any]:
        time.sleep(0.08)
        return {"action": "walk", "confidence": 1.0, "topk": []}


def test_parallel_object_and_action_wall_time() -> None:
    om = _SlowObject()
    am = _SlowAction()
    mgr = AgentManager(
        object_agent=om,
        action_agent=am,
        ocr_agent=None,
        face_pose_agent=None,
        timeout_object_s=2.0,
        timeout_action_s=2.0,
        max_workers=4,
    )
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    packets = [
        FramePacket(
            data=rgb,
            t_mono=0.0,
            source="file",
            frame_id=i,
        )
        for i in range(6)
    ]
    t0 = time.perf_counter()
    _ = mgr.run_parallel(
        frame_rgb=rgb,
        active=frozenset({"object", "action"}),
        action_packets=packets,
    )
    wall_parallel = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = om.detect_instances(rgb)
    _ = am.predict(packets)
    wall_seq = time.perf_counter() - t1
    mgr.close()
    assert wall_parallel < wall_seq * 0.85


def test_timeout_isolation() -> None:
    class _Hang:
        def detect_instances(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
            time.sleep(10.0)
            return []

    mgr = AgentManager(
        object_agent=_Hang(),
        action_agent=None,
        ocr_agent=None,
        face_pose_agent=None,
        timeout_object_s=0.05,
        max_workers=2,
    )
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    out = mgr.run_parallel(frame_rgb=rgb, active=frozenset({"object"}))
    assert out["object"].status == "timeout"
    mgr.close()
