"""YOLOv8n ONNX INT8 object expert for Phase 2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml  # type: ignore[import-untyped]

from src.agents.yolo_postprocess import decode_yolo_predictions, nms_class_aware
from src.utils.artifact_paths import require_file
from src.utils.onnx_benchmark import create_session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_yolo_model_path() -> Path:
    registry_path = _repo_root() / "models" / "registry.json"
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        for model in payload.get("models", []):
            if model.get("id") == "yolov8n":
                artifacts = model.get("artifacts", {})
                int8_meta = artifacts.get("int8_onnx")
                if isinstance(int8_meta, dict) and int8_meta.get("path"):
                    return _repo_root() / str(int8_meta["path"])
    return _repo_root() / "models" / "yolov8n_int8.onnx"


def _load_accessibility_labels() -> list[str]:
    cfg = _repo_root() / "configs" / "datasets" / "object_detection_accessibility.yaml"
    if not cfg.is_file():
        return ["person", "dog", "bicycle", "car"]
    payload = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    classes = payload.get("classes", {}) if isinstance(payload, dict) else {}
    out: list[str] = []
    if isinstance(classes.get("coco_subset"), list):
        out.extend(str(x) for x in classes["coco_subset"])
    if isinstance(classes.get("custom_only"), list):
        out.extend(str(x) for x in classes["custom_only"])
    return out or ["person", "dog", "bicycle", "car"]


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    out[:, 2] = np.clip(out[:, 2], 0, width - 1)
    out[:, 3] = np.clip(out[:, 3], 0, height - 1)
    return out


class ObjectAgent:
    """Object detector with ONNXRuntime preprocessing and postprocessing."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        labels: list[str] | None = None,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        providers: list[str] | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path) if model_path is not None else _resolve_yolo_model_path()
        )
        self.labels = labels or _load_accessibility_labels()
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        if session is not None:
            self._session = session
        else:
            require_file(
                self.model_path,
                model_id="yolov8n",
                hint="INT8 ONNX path is listed under models.registry.json artifacts.",
            )
            self._session = create_session(self.model_path, providers=providers)
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = getattr(input_meta, "shape", None) or []
        batch_dim = shape[0] if len(shape) > 0 else None
        self._static_batch = int(batch_dim) if isinstance(batch_dim, int) else 1

    def preprocess(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize RGB frame to model square input and return NCHW float32."""

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must be HWC RGB")
        h, w = frame_rgb.shape[:2]
        resized = cv2.resize(
            frame_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0).astype(np.float32, copy=False)
        if self._static_batch > 1:
            x = np.repeat(x, self._static_batch, axis=0)
        return x, (h, w)

    def _rescale_boxes(
        self,
        boxes: np.ndarray,
        orig_shape: tuple[int, int],
    ) -> np.ndarray:
        h, w = orig_shape
        sx = float(w) / float(self.input_size)
        sy = float(h) / float(self.input_size)
        out = boxes.copy()
        out[:, [0, 2]] *= sx
        out[:, [1, 3]] *= sy
        return _clip_boxes(out, w, h)

    def detect_instances(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return per-instance records with label/confidence/bbox/count."""

        x, orig_shape = self.preprocess(frame_rgb)
        outputs = self._session.run(None, {self._input_name: x})
        raw = outputs[0]
        boxes, scores, class_ids = decode_yolo_predictions(
            raw,
            conf_threshold=self.conf_threshold,
        )
        if boxes.shape[0] == 0:
            return []
        keep = nms_class_aware(
            boxes,
            scores,
            class_ids,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        boxes = self._rescale_boxes(boxes[keep], orig_shape)
        scores = scores[keep]
        class_ids = class_ids[keep]

        counts: dict[str, int] = {}
        labels: list[str] = []
        for cls in class_ids:
            if int(cls) < len(self.labels):
                name = self.labels[int(cls)]
            else:
                name = f"class_{int(cls)}"
            labels.append(name)
            counts[name] = counts.get(name, 0) + 1

        rows: list[dict[str, Any]] = []
        for i, name in enumerate(labels):
            x1, y1, x2, y2 = boxes[i].tolist()
            rows.append(
                {
                    "label": name,
                    "confidence": float(scores[i]),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "count": int(counts[name]),
                }
            )
        return rows

    def detect(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Alias for standardized per-instance output."""

        return self.detect_instances(frame_rgb)
