from __future__ import annotations

from src.caption.template_engine import SpatialTemplateEngine


def test_template_engine_hazard_first_and_direction_distance() -> None:
    engine = SpatialTemplateEngine(hazard_labels={"car", "stairs"})
    detections = [
        {
            "label": "chair",
            "confidence": 0.7,
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 80.0, "y2": 120.0},
            "count": 1,
        },
        {
            "label": "car",
            "confidence": 0.8,
            "bbox": {"x1": 250.0, "y1": 100.0, "x2": 600.0, "y2": 460.0},
            "count": 1,
        },
    ]
    events = engine.render(detections, frame_width=640, frame_height=480)
    assert len(events) == 2
    assert events[0].priority == "hazard"
    assert "car" in events[0].text
    assert events[1].priority == "info"


def test_template_engine_empty_policy_message() -> None:
    engine = SpatialTemplateEngine(empty_policy="message")
    events = engine.render([], frame_width=640, frame_height=480)
    assert len(events) == 1
    assert "Không có gì đặc biệt" in events[0].text
