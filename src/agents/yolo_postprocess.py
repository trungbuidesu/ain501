"""YOLO ONNX decode and NMS helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and many xyxy boxes."""

    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def nms_class_aware(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Greedy class-aware NMS returning selected indices."""

    keep: list[int] = []
    order = np.argsort(-scores)
    while order.size > 0 and len(keep) < max_detections:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rem = order[1:]
        same_cls = class_ids[rem] == class_ids[i]
        ious = box_iou_xyxy(boxes_xyxy[i], boxes_xyxy[rem])
        suppress = same_cls & (ious > iou_threshold)
        order = rem[~suppress]
    return keep


def decode_yolo_predictions(
    raw_output: Any,
    *,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO-style output into boxes, scores, class_ids arrays."""

    arr = np.asarray(raw_output)
    if arr.ndim == 3:
        # Support both dynamic batch=1 and static batch>1 graphs.
        arr = arr[0]

    # Case A: already postprocessed Nx6 => x1,y1,x2,y2,score,class
    if arr.ndim == 2 and arr.shape[1] == 6:
        boxes = arr[:, :4].astype(np.float32, copy=False)
        scores = arr[:, 4].astype(np.float32, copy=False)
        cls = arr[:, 5].astype(np.int64, copy=False)
        mask = scores >= conf_threshold
        return boxes[mask], scores[mask], cls[mask]

    # Case B: YOLOv8 raw output 84xN or Nx84.
    if arr.ndim != 2:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 6:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    xywh = arr[:, :4].astype(np.float32, copy=False)
    cls_scores = arr[:, 4:].astype(np.float32, copy=False)
    class_ids = np.argmax(cls_scores, axis=1).astype(np.int64)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    mask = scores >= conf_threshold
    boxes = _xywh_to_xyxy(xywh[mask])
    return boxes, scores[mask], class_ids[mask]
