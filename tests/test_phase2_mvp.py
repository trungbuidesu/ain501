from __future__ import annotations

from pathlib import Path

from src.app.phase2_mvp import load_phase2_config, run_phase2
from src.capture.fake_source import FakeFrameSource


def test_load_phase2_config_defaults() -> None:
    cfg = load_phase2_config(Path("configs/phase2_mvp.yaml"))
    assert cfg.conf_threshold > 0
    assert cfg.iou_threshold > 0
    assert cfg.max_detections > 0


def test_run_phase2_with_fake_source(monkeypatch, tmp_path) -> None:
    cfg = load_phase2_config(Path("configs/phase2_mvp.yaml"))
    cfg = type(cfg)(
        capture_config=cfg.capture_config,
        yolo_model_path=tmp_path / "yolov8n_int8.onnx",
        mobilenet_model_path=tmp_path / "mobilenetv3_small_fp32.onnx",
        piper_model_path=tmp_path / "voice.onnx",
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        max_detections=cfg.max_detections,
        use_encoder=False,
        debug_overlay=False,
        reports_dir=tmp_path / "reports",
        empty_policy="silent",
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
        "src.app.phase2_mvp.ObjectAgent",
        lambda *a, **k: _Agent(),
    )
    monkeypatch.setattr(
        "src.app.phase2_mvp.AudioOutputManager",
        lambda *a, **k: _Audio(),
    )

    source = FakeFrameSource(64, 64, 10, pattern="alternate")
    report = run_phase2(cfg, max_frames=8, source_override=source)
    assert report["processed_frames"] >= 8
    assert (cfg.reports_dir / "phase2_latency.json").is_file()
