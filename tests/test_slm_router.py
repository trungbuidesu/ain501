"""Tests for SLM router and fallback heuristics."""

from __future__ import annotations

import numpy as np

from src.router.slm_router import (
    SLMRouter,
    compute_fallback_signals,
    resolve_scene_classifier_path,
)


def test_compute_fallback_signals_no_prev() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    sig = compute_fallback_signals(rgb, None)
    assert sig.motion_diff == 0.0
    assert 0.0 <= sig.edge_density <= 1.0
    assert 0.0 <= sig.skin_ratio <= 1.0


def test_resolve_scene_classifier_path_default() -> None:
    p = resolve_scene_classifier_path()
    assert p.name.endswith(".onnx")


def test_route_high_confidence_uses_model_class() -> None:
    class _In:
        name = "images"

    class MockSession:
        def get_inputs(self):
            return [_In()]

        def run(self, _, feeds):
            logits = np.array([-1.0, -1.0, 5.0, -1.0, -1.0], dtype=np.float32).reshape(
                1, -1
            )
            return [logits]

    scene_map = {
        "static": ["object"],
        "motion": ["object"],
        "text_present": ["object", "ocr"],
        "face_detected": ["object"],
        "scene_change": ["object"],
    }
    router = SLMRouter(
        model_path=None,
        confidence_threshold=0.1,
        scene_to_agents=scene_map,
        session=MockSession(),
        strict_artifacts=False,
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dec = router.route(frame, None)
    assert dec.scene_type == "text_present"
    assert "ocr" in dec.active_agents
    assert dec.used_router_model is True
    assert dec.used_fallback is False


def test_route_low_confidence_uses_fallback() -> None:
    class _In:
        name = "images"

    class MockSession:
        def get_inputs(self):
            return [_In()]

        def run(self, _, feeds):
            logits = np.ones((1, 5), dtype=np.float32) * 0.01
            return [logits]

    scene_map = {
        "static": ["object"],
        "motion": ["object", "action"],
        "text_present": ["object"],
        "face_detected": ["object"],
        "scene_change": ["object"],
    }
    router = SLMRouter(
        model_path=None,
        confidence_threshold=0.99,
        scene_to_agents=scene_map,
        session=MockSession(),
        strict_artifacts=False,
    )
    frame = np.full((80, 80, 3), 200, dtype=np.uint8)
    prev = np.zeros((80, 80, 3), dtype=np.uint8)
    dec = router.route(frame, prev)
    assert dec.used_fallback is True
    assert dec.scene_type in {
        "static",
        "motion",
        "text_present",
        "face_detected",
        "scene_change",
    }
