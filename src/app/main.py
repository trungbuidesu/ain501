"""Accessibility app: capture, detect, template/RAG, TTS, optional Tier 1 agents."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.app.runtime import AppRuntime, build_pipeline_status
from src.caption.template_engine import CaptionEvent
from src.capture.app import _resolve_config_path
from src.capture.config import load_capture_config
from src.capture.paced_loop import pace_after_frame
from src.capture.sources import frame_source_from_config
from src.capture.types import FramePacket
from src.core.hotkeys import start_hotkey_listener, stop_hotkey_listener
from src.detection.change_detector import ChangeDetector
from src.output.audio_output import play_hazard_beep
from src.ui.state import DemoControlState


def _as_mapping(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_print(line: str) -> None:
    """Avoid UnicodeEncodeError on Windows consoles using legacy code pages."""

    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


@dataclass(frozen=True, slots=True)
class AppConfig:
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
    rag_enabled: bool
    rag_model_path: Path
    rag_knowledge_dir: Path
    rag_top_k: int
    rag_min_similarity: float
    rag_dedup_similarity: float
    hotkeys: dict[str, str] = field(default_factory=dict)
    default_verbosity: str = "medium"
    screenshot_dir: Path = field(default_factory=lambda: Path("screenshots"))


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


def _is_spatial_navigation_event(ev: CaptionEvent) -> bool:
    """Template spatial lines use ``priority:label:direction:distance``."""

    if ev.signature.startswith("rag:"):
        return False
    if ev.signature in ("none",):
        return False
    parts = ev.signature.split(":")
    return len(parts) >= 4 and parts[0] in ("hazard", "info")


def filter_caption_events_for_verbosity(
    events: list[CaptionEvent],
    sources: list[str],
    verbosity: str,
) -> tuple[list[CaptionEvent], list[str]]:
    """low: hazard only; medium: hazard + navigation; high: all."""

    if verbosity == "high":
        return list(events), list(sources)
    out_e: list[CaptionEvent] = []
    out_s: list[str] = []
    for ev, src in zip(events, sources, strict=True):
        pr = ev.priority
        if verbosity == "low":
            if pr == "hazard":
                out_e.append(ev)
                out_s.append(src)
            continue
        # medium
        if pr == "hazard" or pr == "navigation":
            out_e.append(ev)
            out_s.append(src)
        elif pr == "info":
            if ev.signature.startswith("rag:"):
                continue
            if _is_spatial_navigation_event(ev):
                out_e.append(ev)
                out_s.append(src)
    return out_e, out_s


def load_app_config(path: Path | str) -> AppConfig:
    p = _resolve_config_path(Path(path))
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("app config root must be mapping")
    root = Path(__file__).resolve().parents[2]
    m = _as_mapping(raw.get("models"))
    d = _as_mapping(raw.get("detector"))
    c = _as_mapping(raw.get("caption"))
    o = _as_mapping(raw.get("output"))
    t1 = _as_mapping(raw.get("tier1"))
    rag_block = _as_mapping(raw.get("rag"))
    hk = _as_mapping(raw.get("hotkeys"))
    hotkeys = {str(k): str(v) for k, v in hk.items()}
    ui = _as_mapping(raw.get("ui"))
    _sd = str(ui.get("screenshot_dir", "screenshots")).strip().rstrip("/\\")
    screenshot_dir = (root / _sd).resolve()

    capture_config = root / str(
        raw.get("capture_config", "configs/capture_pipeline.yaml")
    )
    yolo_model_path = _resolve_under_root(
        root, m.get("yolo_int8"), "models/yolov8n_int8.onnx"
    )
    mobilenet_model_path = _resolve_under_root(
        root, m.get("mobilenet_fp32"), "models/mobilenetv3_small_fp32.onnx"
    )
    piper_model_path = _resolve_under_root(
        root,
        m.get("piper_model_path"),
        "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
    )

    router_block = t1.get("router", {}) if isinstance(t1.get("router"), dict) else {}
    fb_block = (
        router_block.get("fallback", {})
        if isinstance(router_block.get("fallback"), dict)
        else {}
    )
    fallback: dict[str, float] = {}
    for k in (
        "motion_diff_threshold",
        "edge_density_threshold",
        "skin_ratio_threshold",
        "motion_diff_scene_change",
    ):
        if k in fb_block:
            fallback[k] = float(fb_block[k])

    timeouts_block = (
        t1.get("timeouts", {}) if isinstance(t1.get("timeouts"), dict) else {}
    )
    timeouts: dict[str, float] = {
        "object": float(timeouts_block.get("object_s", 2.0)),
        "action": float(timeouts_block.get("action_s", 2.0)),
        "ocr": float(timeouts_block.get("ocr_s", 5.0)),
        "face_pose": float(timeouts_block.get("face_pose_s", 3.0)),
    }

    scene_raw = router_block.get("scene_to_agents", {})
    scene_to_agents = (
        _parse_scene_to_agents(scene_raw)
        if isinstance(scene_raw, dict)
        else _parse_scene_to_agents({})
    )

    movinet_p = t1.get("movinet_int8")
    mediapipe_dir = t1.get("mediapipe_model_dir")

    return AppConfig(
        capture_config=capture_config.resolve(),
        yolo_model_path=yolo_model_path,
        mobilenet_model_path=mobilenet_model_path,
        piper_model_path=piper_model_path,
        conf_threshold=float(d.get("conf_threshold", 0.25)),
        iou_threshold=float(d.get("iou_threshold", 0.45)),
        max_detections=int(d.get("max_detections", 100)),
        use_encoder=bool(_as_mapping(raw.get("encoder")).get("enabled", False)),
        debug_overlay=bool(o.get("debug_overlay", False)),
        reports_dir=(root / str(o.get("reports_dir", "reports/app"))).resolve(),
        empty_policy=str(c.get("empty_policy", "silent")),
        dedup_window_s=float(o.get("dedup_window_s", 2.5)),
        tier1_enabled=bool(t1.get("enabled", False)),
        tier1_strict_artifacts=bool(t1.get("strict_artifacts", True)),
        tier1_object_only=bool(t1.get("object_only", False)),
        tier1_ring_maxlen=int(t1.get("ring_buffer_maxlen", 48)),
        tier1_router_path=(
            _resolve_under_root(
                root,
                (
                    str(router_block.get("model_path"))
                    if router_block.get("model_path")
                    else None
                ),
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
            _resolve_under_root(
                root,
                str(movinet_p) if movinet_p else None,
                "models/movinet_a0_int8.onnx",
            )
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
        rag_enabled=bool(rag_block.get("enabled", False)),
        rag_model_path=_resolve_under_root(
            root,
            str(rag_block.get("model_path")) if rag_block.get("model_path") else None,
            "models/minilm_l6.onnx",
        ),
        rag_knowledge_dir=_resolve_under_root(
            root,
            (
                str(rag_block.get("knowledge_dir"))
                if rag_block.get("knowledge_dir")
                else None
            ),
            "data/knowledge",
        ),
        rag_top_k=int(rag_block.get("top_k", 3)),
        rag_min_similarity=float(rag_block.get("min_similarity", 0.3)),
        rag_dedup_similarity=float(rag_block.get("dedup_similarity", 0.8)),
        hotkeys=hotkeys,
        default_verbosity=str(raw.get("verbosity", "medium")),
        screenshot_dir=screenshot_dir.resolve(),
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


def _run_app_loop(
    cfg: AppConfig,
    *,
    max_frames: int | None,
    source_iter: Any,
    dry_run: bool,
    visual: bool,
    enable_tts: bool,
    verbosity: str,
    hazard_beep: bool,
) -> dict[str, Any]:
    """Drive one frame source until end; ``source_iter`` yields ``FramePacket``."""

    capture_cfg = load_capture_config(cfg.capture_config)
    gate = ChangeDetector(capture_cfg.change_detection)
    runtime = AppRuntime(
        cfg,
        dry_run=dry_run,
        debug_overlay=bool(cfg.debug_overlay and not visual and not dry_run),
    )
    audio = runtime.audio
    overlay = runtime.overlay
    manager = runtime.manager

    demo_control = DemoControlState()
    demo_control.verbosity = verbosity
    demo_control.rag_enabled = cfg.rag_enabled
    demo: Any = None
    if visual:
        from src.ui.demo_window import DemoWindow

        demo = DemoWindow(control=demo_control, screenshot_dir=cfg.screenshot_dir)

    hk_handle: Any = None
    if cfg.hotkeys:
        pair = start_hotkey_listener(cfg.hotkeys, demo_control)
        if pair is not None:
            hk_handle = pair[0]

    stage_detect_ms: list[float] = []
    stage_template_ms: list[float] = []
    stage_audio_enqueue_ms: list[float] = []
    stage_e2e_ms: list[float] = []
    stage_router_ms: list[float] = []
    stage_manager_ms: list[float] = []
    processed = 0
    emitted = 0
    prev = None
    tier1_meta: list[dict[str, Any]] = []
    last_tick = time.monotonic()
    fps_smooth = 0.0

    try:
        for packet in source_iter:
            if packet is None:
                break
            tick = time.perf_counter()
            loop_start = time.monotonic()
            processed += 1
            if max_frames is not None and processed > max_frames:
                break
            if not gate.should_process(prev, packet.data):
                pace_after_frame(loop_start, capture_cfg.timing.target_fps)
                continue
            prev = packet.data.copy()
            emitted += 1

            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            if dt > 1e-6:
                inst = 1.0 / dt
                fps_smooth = 0.85 * fps_smooth + 0.15 * inst if fps_smooth > 0 else inst

            runtime.set_rag_enabled(
                demo_control.rag_enabled if demo is not None else cfg.rag_enabled
            )

            result = runtime.process_frame(packet, tick_start=tick)

            if result.router_ms is not None:
                stage_router_ms.append(result.router_ms)
            if result.manager_ms is not None:
                stage_manager_ms.append(result.manager_ms)
            if result.detect_ms is not None:
                stage_detect_ms.append(result.detect_ms)
            stage_template_ms.append(result.template_ms)
            stage_e2e_ms.append(result.e2e_ms)

            verb = demo_control.verbosity if demo is not None else verbosity
            is_muted = (not enable_tts) or (
                demo_control.muted if demo is not None else False
            )
            status = build_pipeline_status(
                fps=fps_smooth,
                latency_ms=result.e2e_ms,
                result=result,
                verbosity=verb,  # type: ignore[arg-type]
                is_muted=is_muted,
            )

            if result.routing is not None:
                tier1_meta.append(
                    {
                        "scene_type": result.routing.scene_type,
                        "confidence": result.routing.confidence,
                        "active_agents": sorted(result.active_agents),
                        "used_fallback": result.routing.used_fallback,
                    }
                )

            t2 = time.perf_counter()
            ev_full = list(result.events)
            src_full = result.caption_sources
            ev_filt, src_filt = filter_caption_events_for_verbosity(
                ev_full, src_full, verb
            )
            for i, event in enumerate(ev_filt):
                _src = src_filt[i] if i < len(src_filt) else "template"
                if dry_run:
                    cap = (
                        f"[caption] {event.text} source={_src} "
                        f"priority={event.priority}"
                    )
                    _safe_print(cap)
                elif not is_muted and enable_tts:
                    if event.priority == "hazard" and hazard_beep:
                        play_hazard_beep()
                    ok = audio.enqueue(event.text, event.priority, event.signature)
                    if not ok and not audio.is_enabled:
                        audio.speak_print_fallback(event.text)
                overlay.show_text(event.text)

            if demo is not None and demo.running:
                cap_text = "\n".join(e.text for e in ev_filt[:2]) or ""
                demo.show_frame(
                    packet.data,
                    result.detections,
                    result,
                    cap_text,
                    status,
                )
                if not demo.running:
                    break

            stage_audio_enqueue_ms.append((time.perf_counter() - t2) * 1000.0)

            pace_after_frame(loop_start, capture_cfg.timing.target_fps)
    finally:
        runtime.close()
        if demo is not None:
            demo.close()
        if hk_handle is not None:
            stop_hotkey_listener(hk_handle)

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
            float(statistics.median(stage_template_ms)) if stage_template_ms else None
        ),
        "audio_enqueue_ms_p50": (
            float(statistics.median(stage_audio_enqueue_ms))
            if stage_audio_enqueue_ms
            else None
        ),
        "tier1_enabled": cfg.tier1_enabled,
        "rag_enabled": cfg.rag_enabled,
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

    out = cfg.reports_dir / "app_latency.json"
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


def _iter_from_source(source: Any) -> Any:
    while True:
        packet = source.read()
        if packet is None:
            break
        yield packet


def run_app(
    cfg: AppConfig,
    *,
    max_frames: int | None = None,
    source_override: Any | None = None,
    dry_run: bool = False,
    visual: bool = False,
    enable_tts: bool = True,
    use_threaded_pipeline: bool = False,
    verbosity: str = "medium",
    hazard_beep: bool = True,
) -> dict[str, Any]:
    if use_threaded_pipeline:
        return _run_app_threaded(
            cfg,
            max_frames=max_frames,
            source_override=source_override,
            dry_run=dry_run,
            visual=visual,
            enable_tts=enable_tts,
            verbosity=verbosity,
            hazard_beep=hazard_beep,
        )

    capture_cfg = load_capture_config(cfg.capture_config)
    source = (
        source_override
        if source_override is not None
        else frame_source_from_config(capture_cfg)
    )
    try:
        return _run_app_loop(
            cfg,
            max_frames=max_frames,
            source_iter=_iter_from_source(source),
            dry_run=dry_run,
            visual=visual,
            enable_tts=enable_tts,
            verbosity=verbosity,
            hazard_beep=hazard_beep,
        )
    finally:
        source.close()


def _run_app_threaded(
    cfg: AppConfig,
    *,
    max_frames: int | None,
    source_override: Any | None,
    dry_run: bool,
    visual: bool,
    enable_tts: bool,
    verbosity: str,
    hazard_beep: bool,
) -> dict[str, Any]:
    """Capture on a worker thread; process frames on the main thread."""

    capture_cfg = load_capture_config(cfg.capture_config)
    source = (
        source_override
        if source_override is not None
        else frame_source_from_config(capture_cfg)
    )
    q: queue.Queue[FramePacket | None] = queue.Queue(maxsize=2)
    stop = threading.Event()
    exc_holder: list[BaseException] = []

    def capture_worker() -> None:
        n = 0
        try:
            while not stop.is_set():
                packet = source.read()
                if packet is None:
                    q.put(None)
                    return
                n += 1
                if max_frames is not None and n > max_frames:
                    q.put(None)
                    return
                q.put(packet)
        except BaseException as e:
            exc_holder.append(e)
            try:
                q.put(None)
            except Exception:
                pass
        finally:
            try:
                source.close()
            except Exception:
                pass

    ct = threading.Thread(target=capture_worker, name="CaptureThread", daemon=True)
    ct.start()

    def threaded_iter() -> Any:
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    try:
        return _run_app_loop(
            cfg,
            max_frames=None,
            source_iter=threaded_iter(),
            dry_run=dry_run,
            visual=visual,
            enable_tts=enable_tts,
            verbosity=verbosity,
            hazard_beep=hazard_beep,
        )
    finally:
        stop.set()
        ct.join(timeout=5.0)
        if exc_holder:
            raise exc_holder[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    p.add_argument(
        "--capture-config",
        type=Path,
        default=None,
        help="Override capture pipeline YAML (e.g. configs/capture_webcam.yaml).",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--object-only",
        action="store_true",
        help="Tier1 debug: run object detector only (skip router and other experts).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print captions to console; do not run Piper TTS.",
    )
    p.add_argument(
        "--visual",
        action="store_true",
        help="OpenCV demo window (overrides ui.enabled in app.yaml).",
    )
    p.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable speech (useful with --visual).",
    )
    p.add_argument(
        "--tts",
        action="store_true",
        help="Keep TTS enabled (default on unless --no-tts).",
    )
    p.add_argument(
        "--threaded-pipeline",
        action="store_true",
        help="Capture frames on a worker thread; process on the main thread.",
    )
    p.add_argument(
        "--verbosity",
        choices=("low", "medium", "high"),
        default=None,
        help="Caption filter (default: verbosity in app.yaml).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_app_config(args.config)
    if args.capture_config is not None:
        cfg = replace(
            cfg,
            capture_config=_resolve_config_path(args.capture_config).resolve(),
        )
    if args.object_only:
        cfg = replace(cfg, tier1_object_only=True)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ui_block = raw.get("ui") if isinstance(raw.get("ui"), dict) else {}
    visual = bool(args.visual or ui_block.get("enabled", False))
    enable_tts = not bool(args.no_tts)
    if args.tts:
        enable_tts = True

    verb = (
        str(args.verbosity)
        if args.verbosity is not None
        else str(cfg.default_verbosity)
    )

    report = run_app(
        cfg,
        max_frames=args.max_frames,
        dry_run=bool(args.dry_run),
        visual=visual,
        enable_tts=enable_tts,
        use_threaded_pipeline=bool(args.threaded_pipeline),
        verbosity=verb,
        hazard_beep=True,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
