from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.app.main import load_app_config, run_app
from src.capture.fake_source import FakeFrameSource


def test_load_app_config_defaults() -> None:
    cfg = load_app_config(Path("configs/app.yaml"))
    assert cfg.conf_threshold > 0
    assert cfg.iou_threshold > 0
    assert cfg.max_detections > 0


def test_run_app_with_fake_source(monkeypatch, tmp_path) -> None:
    cfg = load_app_config(Path("configs/app.yaml"))
    cfg = replace(
        cfg,
        yolo_model_path=tmp_path / "yolov8n_int8.onnx",
        mobilenet_model_path=tmp_path / "mobilenetv3_small_fp32.onnx",
        piper_model_path=tmp_path / "voice.onnx",
        use_encoder=False,
        debug_overlay=False,
        reports_dir=tmp_path / "reports",
        dedup_window_s=0.1,
    )

    class _Agent:
        def detect_instances(self, _frame):
            return [
                {
                    "label": "car",
                    "confidence": 0.9,
                    "bbox": {"x1": 20.0, "y1": 10.0, "x2": 120.0, "y2": 80.0},
                    "count": 1,
                }
            ]

    class _Audio:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def start(self):
            return

        def stop(self):
            return

        def enqueue(self, text, priority, signature):
            self.rows.append((text, priority, signature))
            return True

    monkeypatch.setattr(
        "src.app.runtime.ObjectAgent",
        lambda *a, **k: _Agent(),
    )
    monkeypatch.setattr(
        "src.app.runtime.AudioOutputManager",
        lambda *a, **k: _Audio(),
    )

    source = FakeFrameSource(64, 64, 10, pattern="alternate")
    report = run_app(cfg, max_frames=8, source_override=source)
    assert report["processed_frames"] >= 8
    assert (cfg.reports_dir / "app_latency.json").is_file()
