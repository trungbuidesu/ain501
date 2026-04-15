# Phase 2 MVP: Object to Speech

## Run

```text
python -m src.app.phase2_mvp --config configs/phase2_mvp.yaml
```

Optional quick smoke run:

```text
python -m src.app.phase2_mvp --config configs/phase2_mvp.yaml --max-frames 120
```

## Main Config Knobs

- `capture_config`: path to capture/change-detection YAML.
- `models.yolo_int8`: YOLOv8n ONNX INT8 artifact path.
- `models.mobilenet_fp32`: MobileNetV3-Small ONNX path for feature extraction.
- `models.piper_model_path`: Piper ONNX voice file.
- `detector.conf_threshold`, `detector.iou_threshold`, `detector.max_detections`.
- `caption.empty_policy`: `silent` or fallback message mode.
- `output.dedup_window_s`: suppress repeated scene text in this window.
- `output.debug_overlay`: optional QLabel caption display.

## Acceptance Checklist

- App reads frames from configured capture source.
- Object detection emits normalized entries with labels, confidence, bbox, and count.
- Template engine prioritizes hazard objects before informational objects.
- TTS queue respects interrupt policy (higher priority preempts lower priority).
- End-to-end latency report is written to `reports/phase2/phase2_latency.json`.
