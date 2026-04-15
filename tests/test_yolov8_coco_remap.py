"""Tests for COCO→subset remap mAP (no Ultralytics required for core metrics)."""

from __future__ import annotations

import numpy as np

from src.training.eval.yolov8_coco_remap import (
    ImageEval,
    average_precision_one_class,
    build_coco_id_to_subset_id,
    coco80_index,
    map50_macro,
)
from src.training.scripts.eval_yolov8_accessibility_remap import resolve_predict_device


def test_coco80_dog_index() -> None:
    assert coco80_index("dog") == 16
    assert coco80_index("car") == 2


def test_build_map_matches_examples() -> None:
    names = [
        "person",
        "dog",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "traffic light",
        "stop sign",
        "bench",
        "chair",
        "backpack",
        "handbag",
        "suitcase",
        "umbrella",
    ]
    m = build_coco_id_to_subset_id(names)
    assert m[16] == 1  # dog
    assert m[2] == 3  # car


def test_perfect_single_image_ap() -> None:
    s = ImageEval(
        gt_cls=np.array([0], dtype=np.int64),
        gt_xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        pred_cls=np.array([0], dtype=np.int64),
        pred_conf=np.array([0.99]),
        pred_xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
    )
    ap, n = average_precision_one_class([s], 0, 0.5)
    assert n == 1
    assert ap > 0.99


def test_map50_macro_two_classes() -> None:
    samples = [
        ImageEval(
            gt_cls=np.array([0], dtype=np.int64),
            gt_xyxy=np.array([[0, 0, 10, 10]]),
            pred_cls=np.array([0], dtype=np.int64),
            pred_conf=np.array([0.9]),
            pred_xyxy=np.array([[0, 0, 10, 10]]),
        ),
        ImageEval(
            gt_cls=np.array([1], dtype=np.int64),
            gt_xyxy=np.array([[5, 5, 20, 20]]),
            pred_cls=np.array([1], dtype=np.int64),
            pred_conf=np.array([0.8]),
            pred_xyxy=np.array([[5, 5, 20, 20]]),
        ),
    ]
    m_ap, _, _ = map50_macro(samples, num_classes=2, iou_threshold=0.5)
    assert m_ap > 0.99


def test_resolve_predict_device_keeps_cpu() -> None:
    assert resolve_predict_device("cpu") == "cpu"


def test_resolve_predict_device_falls_back_from_xpu() -> None:
    assert resolve_predict_device("xpu") == "cpu"
