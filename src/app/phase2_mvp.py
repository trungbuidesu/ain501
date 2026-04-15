"""Phase 2 MVP: capture, detect, template, and speak via Piper."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.agents.object_agent import ObjectAgent
from src.caption.template_engine import SpatialTemplateEngine
from src.capture.app import _resolve_config_path
from src.capture.config import load_capture_config
from src.capture.paced_loop import pace_after_frame
from src.capture.sources import frame_source_from_config
from src.detection.change_detector import ChangeDetector
from src.encoder.visual_encoder import VisualEncoder
from src.output.audio_output import AudioOutputManager
from src.output.debug_overlay import CaptionDebugOverlay

DEBUG_LOG_PATH = Path("debug-fe6f6f.log")
DEBUG_SESSION_ID = "fe6f6f"


def _debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class Phase2Config:
    capture_config: Path
    yolo_model_path: Path
    mobilenet_model_path: Path
    piper_model_path: Path
    conf_threshold: float
    iou_threshold: float
    max_detections: int
    use_encoder: bool
    debug_overlay: bool
    reports_dir: Path
    empty_policy: str
    dedup_window_s: float


def load_phase2_config(path: Path | str) -> Phase2Config:
    p = _resolve_config_path(Path(path))
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("phase2 config root must be mapping")
    root = Path(__file__).resolve().parents[2]
    m = raw.get("models", {})
    d = raw.get("detector", {})
    c = raw.get("caption", {})
    o = raw.get("output", {})
    capture_config = root / str(
        raw.get("capture_config", "configs/capture_pipeline.yaml")
    )
    yolo_model_path = root / str(m.get("yolo_int8", "models/yolov8n_int8.onnx"))
    mobilenet_model_path = root / str(
        m.get("mobilenet_fp32", "models/mobilenetv3_small_fp32.onnx")
    )
    piper_model_path = root / str(
        m.get(
            "piper_model_path",
            "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
        )
    )
    return Phase2Config(
        capture_config=capture_config.resolve(),
        yolo_model_path=yolo_model_path.resolve(),
        mobilenet_model_path=mobilenet_model_path.resolve(),
        piper_model_path=piper_model_path.resolve(),
        conf_threshold=float(d.get("conf_threshold", 0.25)),
        iou_threshold=float(d.get("iou_threshold", 0.45)),
        max_detections=int(d.get("max_detections", 100)),
        use_encoder=bool(raw.get("encoder", {}).get("enabled", False)),
        debug_overlay=bool(o.get("debug_overlay", False)),
        reports_dir=(root / str(o.get("reports_dir", "reports/phase2"))).resolve(),
        empty_policy=str(c.get("empty_policy", "silent")),
        dedup_window_s=float(o.get("dedup_window_s", 2.5)),
    )


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(len(xs) - 1, lo + 1)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def run_phase2(
    cfg: Phase2Config,
    *,
    max_frames: int | None = None,
    source_override: Any | None = None,
) -> dict[str, Any]:
    run_id = f"phase2_{int(time.time() * 1000)}"
    capture_cfg = load_capture_config(cfg.capture_config)
    source = (
        source_override
        if source_override is not None
        else frame_source_from_config(capture_cfg)
    )
    gate = ChangeDetector(capture_cfg.change_detection)
    agent = ObjectAgent(
        model_path=cfg.yolo_model_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        max_detections=cfg.max_detections,
    )
    engine = SpatialTemplateEngine(empty_policy=cfg.empty_policy)
    overlay = CaptionDebugOverlay(cfg.debug_overlay)
    audio = AudioOutputManager(
        model_path=cfg.piper_model_path,
        output_dir=cfg.reports_dir / "audio",
        dedup_window_s=cfg.dedup_window_s,
        debug_run_id=run_id,
    )
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H5_config_artifact_paths",
        location="src/app/phase2_mvp.py:151",
        message="Phase2 runtime artifact paths and flags",
        data={
            "yolo_model_path": str(cfg.yolo_model_path),
            "yolo_exists": cfg.yolo_model_path.exists(),
            "mobilenet_model_path": str(cfg.mobilenet_model_path),
            "mobilenet_exists": cfg.mobilenet_model_path.exists(),
            "piper_model_path": str(cfg.piper_model_path),
            "piper_model_exists": cfg.piper_model_path.exists(),
            "max_frames": max_frames,
        },
    )
    # endregion
    audio.start()
    encoder = VisualEncoder(cfg.mobilenet_model_path) if cfg.use_encoder else None

    stage_detect_ms: list[float] = []
    stage_template_ms: list[float] = []
    stage_audio_enqueue_ms: list[float] = []
    stage_e2e_ms: list[float] = []
    processed = 0
    emitted = 0
    prev = None
    try:
        while True:
            tick = time.perf_counter()
            loop_start = time.monotonic()
            packet = source.read()
            if packet is None:
                break
            processed += 1
            if max_frames is not None and processed > max_frames:
                break
            if not gate.should_process(prev, packet.data):
                pace_after_frame(loop_start, capture_cfg.timing.target_fps)
                continue
            prev = packet.data.copy()
            emitted += 1

            if encoder is not None:
                _ = encoder.encode_packet(packet)

            t0 = time.perf_counter()
            detections = agent.detect_instances(packet.data)
            stage_detect_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            events = engine.render(
                detections,
                frame_width=packet.data.shape[1],
                frame_height=packet.data.shape[0],
            )
            stage_template_ms.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            for event in events:
                audio.enqueue(event.text, event.priority, event.signature)
                overlay.show_text(event.text)
            stage_audio_enqueue_ms.append((time.perf_counter() - t2) * 1000.0)

            stage_e2e_ms.append((time.perf_counter() - tick) * 1000.0)
            pace_after_frame(loop_start, capture_cfg.timing.target_fps)
    finally:
        source.close()
        audio.stop()
        overlay.close()

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    detect_sorted = sorted(stage_detect_ms)
    e2e_sorted = sorted(stage_e2e_ms)
    report = {
        "processed_frames": processed,
        "emitted_frames": emitted,
        "detect_ms_p50": (
            float(statistics.median(stage_detect_ms)) if stage_detect_ms else None
        ),
        "detect_ms_p95": (
            float(_percentile(detect_sorted, 95)) if detect_sorted else None
        ),
        "e2e_ms_p50": float(statistics.median(stage_e2e_ms)) if stage_e2e_ms else None,
        "e2e_ms_p95": float(_percentile(e2e_sorted, 95)) if e2e_sorted else None,
        "template_ms_p50": (
            float(statistics.median(stage_template_ms))
            if stage_template_ms
            else None
        ),
        "audio_enqueue_ms_p50": (
            float(statistics.median(stage_audio_enqueue_ms))
            if stage_audio_enqueue_ms
            else None
        ),
    }
    out = cfg.reports_dir / "phase2_latency.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/phase2_mvp.yaml"))
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_phase2_config(args.config)
    report = run_phase2(cfg, max_frames=args.max_frames)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
