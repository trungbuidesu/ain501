"""Phase 2 MVP: capture, detect, template, and speak via Piper.

Tier 1: optional SLM router + parallel agents (action, OCR, face/pose).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from src.agents.action_agent import ActionAgent
from src.agents.agent_manager import AgentManager
from src.agents.face_pose_agent import FacePoseAgent
from src.agents.object_agent import ObjectAgent
from src.agents.ocr_agent import OcrAgent
from src.caption.template_engine import SpatialTemplateEngine
from src.caption.tier1_caption import merge_tier1_caption_events
from src.capture.app import _resolve_config_path
from src.capture.config import load_capture_config
from src.capture.paced_loop import pace_after_frame
from src.capture.ring_buffer import RingBuffer
from src.capture.sources import frame_source_from_config
from src.detection.change_detector import ChangeDetector
from src.encoder.visual_encoder import VisualEncoder
from src.output.audio_output import AudioOutputManager
from src.output.debug_overlay import CaptionDebugOverlay
from src.router.slm_router import SLMRouter


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
    tier1_enabled: bool
    tier1_strict_artifacts: bool
    tier1_object_only: bool
    tier1_ring_maxlen: int
    tier1_router_path: Path | None
    tier1_router_confidence: float
    tier1_router_image_size: int
    tier1_scene_to_agents: dict[str, frozenset[str]]
    tier1_fallback: dict[str, float]
    tier1_movinet_path: Path | None
    tier1_action_clip_frames: int
    tier1_ocr_confidence: float
    tier1_ocr_grid: int
    tier1_mediapipe_model_dir: Path | None
    tier1_timeouts: dict[str, float]


def _resolve_under_root(root: Path, p: str | None, default: str) -> Path:
    if not p:
        return (root / default).resolve()
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


def _parse_scene_to_agents(raw: dict[str, Any]) -> dict[str, frozenset[str]]:
    base_default = frozenset({"object"})
    names = ("static", "motion", "text_present", "face_detected", "scene_change")
    out: dict[str, frozenset[str]] = {}
    for n in names:
        key = str(n)
        if key in raw and isinstance(raw[key], (list, tuple)):
            out[key] = frozenset(str(x) for x in raw[key])
        else:
            out[key] = base_default
    return out


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
    t1 = raw.get("tier1", {}) if isinstance(raw.get("tier1"), dict) else {}

    capture_config = root / str(
        raw.get("capture_config", "configs/capture_pipeline.yaml")
    )
    yolo_model_path = _resolve_under_root(root, m.get("yolo_int8"), "models/yolov8n_int8.onnx")
    mobilenet_model_path = _resolve_under_root(
        root, m.get("mobilenet_fp32"), "models/mobilenetv3_small_fp32.onnx"
    )
    piper_model_path = _resolve_under_root(
        root,
        m.get("piper_model_path"),
        "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
    )

    router_block = t1.get("router", {}) if isinstance(t1.get("router"), dict) else {}
    fb_block = router_block.get("fallback", {}) if isinstance(router_block.get("fallback"), dict) else {}
    fallback: dict[str, float] = {}
    for k in (
        "motion_diff_threshold",
        "edge_density_threshold",
        "skin_ratio_threshold",
        "motion_diff_scene_change",
    ):
        if k in fb_block:
            fallback[k] = float(fb_block[k])

    timeouts_block = t1.get("timeouts", {}) if isinstance(t1.get("timeouts"), dict) else {}
    timeouts: dict[str, float] = {
        "object": float(timeouts_block.get("object_s", 2.0)),
        "action": float(timeouts_block.get("action_s", 2.0)),
        "ocr": float(timeouts_block.get("ocr_s", 5.0)),
        "face_pose": float(timeouts_block.get("face_pose_s", 3.0)),
    }

    scene_raw = router_block.get("scene_to_agents", {})
    scene_to_agents = (
        _parse_scene_to_agents(scene_raw) if isinstance(scene_raw, dict) else _parse_scene_to_agents({})
    )

    movinet_p = t1.get("movinet_int8")
    mediapipe_dir = t1.get("mediapipe_model_dir")

    return Phase2Config(
        capture_config=capture_config.resolve(),
        yolo_model_path=yolo_model_path,
        mobilenet_model_path=mobilenet_model_path,
        piper_model_path=piper_model_path,
        conf_threshold=float(d.get("conf_threshold", 0.25)),
        iou_threshold=float(d.get("iou_threshold", 0.45)),
        max_detections=int(d.get("max_detections", 100)),
        use_encoder=bool(raw.get("encoder", {}).get("enabled", False)),
        debug_overlay=bool(o.get("debug_overlay", False)),
        reports_dir=(root / str(o.get("reports_dir", "reports/phase2"))).resolve(),
        empty_policy=str(c.get("empty_policy", "silent")),
        dedup_window_s=float(o.get("dedup_window_s", 2.5)),
        tier1_enabled=bool(t1.get("enabled", False)),
        tier1_strict_artifacts=bool(t1.get("strict_artifacts", True)),
        tier1_object_only=bool(t1.get("object_only", False)),
        tier1_ring_maxlen=int(t1.get("ring_buffer_maxlen", 48)),
        tier1_router_path=(
            _resolve_under_root(
                root,
                str(router_block.get("model_path")) if router_block.get("model_path") else None,
                "models/scene_classifier_int8.onnx",
            )
            if t1.get("enabled") and not t1.get("object_only")
            else None
        ),
        tier1_router_confidence=float(router_block.get("confidence_threshold", 0.45)),
        tier1_router_image_size=int(router_block.get("image_size", 160)),
        tier1_scene_to_agents=scene_to_agents,
        tier1_fallback=fallback,
        tier1_movinet_path=(
            _resolve_under_root(root, str(movinet_p) if movinet_p else None, "models/movinet_a0_int8.onnx")
            if t1.get("enabled") and not t1.get("object_only")
            else None
        ),
        tier1_action_clip_frames=int(t1.get("action_clip_frames", 6)),
        tier1_ocr_confidence=float(t1.get("ocr_confidence", 0.5)),
        tier1_ocr_grid=int(t1.get("ocr_grid", 2)),
        tier1_mediapipe_model_dir=(
            _resolve_under_root(
                root,
                str(mediapipe_dir) if mediapipe_dir else None,
                "reports/prebuilt_week5/_mediapipe_models",
            )
            if t1.get("enabled") and not t1.get("object_only")
            else None
        ),
        tier1_timeouts=timeouts,
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
    capture_cfg = load_capture_config(cfg.capture_config)
    source = (
        source_override
        if source_override is not None
        else frame_source_from_config(capture_cfg)
    )
    gate = ChangeDetector(capture_cfg.change_detection)
    object_agent = ObjectAgent(
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
    )
    audio.start()
    encoder = VisualEncoder(cfg.mobilenet_model_path) if cfg.use_encoder else None

    router: SLMRouter | None = None
    ring: RingBuffer | None = None
    action_agent: ActionAgent | None = None
    ocr_agent: OcrAgent | None = None
    face_pose_agent: FacePoseAgent | None = None
    manager: AgentManager | None = None

    if cfg.tier1_enabled and not cfg.tier1_object_only:
        strict = cfg.tier1_strict_artifacts
        scene_path = cfg.tier1_router_path
        router = SLMRouter(
            scene_path,
            confidence_threshold=cfg.tier1_router_confidence,
            image_size=cfg.tier1_router_image_size,
            scene_to_agents={k: list(v) for k, v in cfg.tier1_scene_to_agents.items()},
            fallback=cfg.tier1_fallback,
            strict_artifacts=strict,
        )
        ring = RingBuffer(cfg.tier1_ring_maxlen)
        action_agent = ActionAgent(
            cfg.tier1_movinet_path,
            input_frames=cfg.tier1_action_clip_frames,
            strict_artifacts=strict,
        )
        ocr_agent = OcrAgent(
            confidence_threshold=cfg.tier1_ocr_confidence,
            grid=cfg.tier1_ocr_grid,
        )
        face_pose_agent = FacePoseAgent(
            model_dir=cfg.tier1_mediapipe_model_dir,
            strict_artifacts=False,
        )
        manager = AgentManager(
            object_agent=object_agent,
            action_agent=action_agent,
            ocr_agent=ocr_agent,
            face_pose_agent=face_pose_agent,
            timeout_object_s=cfg.tier1_timeouts["object"],
            timeout_action_s=cfg.tier1_timeouts["action"],
            timeout_ocr_s=cfg.tier1_timeouts["ocr"],
            timeout_face_pose_s=cfg.tier1_timeouts["face_pose"],
        )

    stage_detect_ms: list[float] = []
    stage_template_ms: list[float] = []
    stage_audio_enqueue_ms: list[float] = []
    stage_e2e_ms: list[float] = []
    stage_router_ms: list[float] = []
    stage_manager_ms: list[float] = []
    processed = 0
    emitted = 0
    prev = None
    prev_processed_rgb: np.ndarray | None = None
    tier1_meta: list[dict[str, Any]] = []

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

            routing = None
            active: frozenset[str] = frozenset({"object"})
            if cfg.tier1_enabled and not cfg.tier1_object_only and router is not None:
                t_r0 = time.perf_counter()
                routing = router.route(packet.data, prev_processed_rgb)
                stage_router_ms.append((time.perf_counter() - t_r0) * 1000.0)
                active = routing.active_agents
                prev_processed_rgb = packet.data.copy()
            elif cfg.tier1_enabled and cfg.tier1_object_only:
                active = frozenset({"object"})

            if ring is not None:
                ring.push(packet)

            action_packets = None
            if (
                ring is not None
                and "action" in active
                and action_agent is not None
                and manager is not None
            ):
                packets_list = ring.as_list()
                if len(packets_list) >= 1:
                    action_packets = packets_list[-action_agent.input_frames :]

            t_mgr0 = time.perf_counter()
            if cfg.tier1_enabled and not cfg.tier1_object_only and manager is not None:
                agent_results = manager.run_parallel(
                    frame_rgb=packet.data,
                    active=active,
                    action_packets=action_packets,
                )
                dets = agent_results.get("object")
                if dets is not None and dets.status == "ok":
                    detections = dets.payload
                elif "object" in active:
                    detections = object_agent.detect_instances(packet.data)
                else:
                    detections = []
                stage_manager_ms.append((time.perf_counter() - t_mgr0) * 1000.0)

                t0 = time.perf_counter()
                obj_events = engine.render(
                    detections,
                    frame_width=packet.data.shape[1],
                    frame_height=packet.data.shape[0],
                )
                stage_detect_ms.append((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                events = merge_tier1_caption_events(obj_events, agent_results)
                stage_template_ms.append((time.perf_counter() - t1) * 1000.0)

                if routing is not None:
                    tier1_meta.append(
                        {
                            "scene_type": routing.scene_type,
                            "confidence": routing.confidence,
                            "active_agents": sorted(active),
                            "used_fallback": routing.used_fallback,
                        }
                    )
            else:
                t0 = time.perf_counter()
                detections = object_agent.detect_instances(packet.data)
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
        if manager is not None:
            manager.close()

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    detect_sorted = sorted(stage_detect_ms)
    e2e_sorted = sorted(stage_e2e_ms)
    report: dict[str, Any] = {
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
        "tier1_enabled": cfg.tier1_enabled,
    }
    if stage_router_ms:
        rs = sorted(stage_router_ms)
        report["router_ms_p50"] = float(statistics.median(stage_router_ms))
        report["router_ms_p95"] = float(_percentile(rs, 95))
    if stage_manager_ms:
        ms = sorted(stage_manager_ms)
        report["manager_ms_p50"] = float(statistics.median(stage_manager_ms))
        report["manager_ms_p95"] = float(_percentile(ms, 95))
    if tier1_meta:
        report["tier1_routing_samples"] = tier1_meta[:50]

    out = cfg.reports_dir / "phase2_latency.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if cfg.tier1_enabled:
        phase3_dir = Path(__file__).resolve().parents[2] / "reports" / "phase3"
        phase3_dir.mkdir(parents=True, exist_ok=True)
        tier1_report = {
            "tier1": True,
            "strict_artifacts": cfg.tier1_strict_artifacts,
            "latency": report,
            "agent_manager_stats": (
                {
                    "success": manager.stats.success,
                    "timeout": manager.stats.timeout,
                    "error": manager.stats.error,
                }
                if manager is not None
                else None
            ),
        }
        (phase3_dir / "tier1_benchmark.json").write_text(
            json.dumps(tier1_report, indent=2) + "\n",
            encoding="utf-8",
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/phase2_mvp.yaml"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--object-only",
        action="store_true",
        help="Tier1 debug: run object detector only (skip router and other experts).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_phase2_config(args.config)
    if args.object_only:
        cfg = replace(cfg, tier1_object_only=True)
    report = run_phase2(cfg, max_frames=args.max_frames)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
