"""MoViNet action agent tests (mock ONNX)."""

from __future__ import annotations

import numpy as np

from src.agents.action_agent import ActionAgent
from src.capture.types import FramePacket


def test_action_agent_predict_mock_session() -> None:
    class _In:
        name = "input"

    logits = np.array([0.0] * 12, dtype=np.float32)
    logits[3] = 5.0

    class MockSession:
        def get_inputs(self):
            return [_In()]

        def run(self, _, feeds):
            return [logits.reshape(1, -1)]

    agent = ActionAgent(
        model_path="/nonexistent.onnx",
        strict_artifacts=False,
        session=MockSession(),
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    packets = [
        FramePacket(data=rgb, t_mono=0.0, source="file", frame_id=i) for i in range(6)
    ]
    out = agent.predict(packets)
    assert "action" in out
    assert out["confidence"] > 0.0
    assert isinstance(out["topk"], list)
