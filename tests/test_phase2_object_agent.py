from __future__ import annotations

import numpy as np

from src.agents.object_agent import ObjectAgent


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeYoloSession:
    def __init__(self, raw):
        self.raw = raw

    def get_inputs(self):
        return [_FakeInput("images")]

    def run(self, _outs, _feeds):
        return [self.raw]


def test_object_agent_detect_two_person_one_car() -> None:
    # x1, y1, x2, y2, score, class_id
    raw = np.array(
        [
            [
                [40, 40, 120, 220, 0.95, 0],  # person
                [180, 50, 260, 210, 0.90, 0],  # person
                [300, 120, 430, 260, 0.88, 3],  # car
            ]
        ],
        dtype=np.float32,
    )
    labels = ["person", "dog", "bicycle", "car"]
    agent = ObjectAgent(
        session=_FakeYoloSession(raw),
        labels=labels,
        conf_threshold=0.2,
        iou_threshold=0.5,
        input_size=640,
    )
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    out = agent.detect_instances(image)
    assert len(out) == 3
    persons = [d for d in out if d["label"] == "person"]
    cars = [d for d in out if d["label"] == "car"]
    assert len(persons) == 2
    assert len(cars) == 1
    assert all(d["count"] == 2 for d in persons)
    assert cars[0]["count"] == 1


def test_object_agent_applies_conf_threshold() -> None:
    raw = np.array(
        [[[40, 40, 100, 100, 0.15, 0], [60, 60, 140, 140, 0.91, 0]]],
        dtype=np.float32,
    )
    agent = ObjectAgent(
        session=_FakeYoloSession(raw),
        labels=["person"],
        conf_threshold=0.2,
        iou_threshold=0.5,
        input_size=640,
    )
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    out = agent.detect_instances(image)
    assert len(out) == 1
    assert out[0]["confidence"] > 0.9
