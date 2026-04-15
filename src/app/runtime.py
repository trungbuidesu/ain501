"""Per-frame processing state for the accessibility app (import cycle safe)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents.action_agent import ActionAgent
from src.agents.agent_manager import AgentManager
from src.agents.face_pose_agent import FacePoseAgent
from src.agents.object_agent import ObjectAgent
from src.agents.ocr_agent import OcrAgent
from src.caption.orchestrator import CaptionOrchestrator
from src.caption.template_engine import CaptionEvent, SpatialTemplateEngine
from src.capture.types import FramePacket
from src.core.types import PipelineStatus, VerbosityLevel
from src.encoder.visual_encoder import VisualEncoder
from src.output.audio_output import AudioOutputManager
from src.output.debug_overlay import CaptionDebugOverlay
from src.rag.knowledge_base import KnowledgeBase
from src.router.slm_router import RoutingDecision, SLMRouter


@dataclass
class FrameProcessResult:
    """Outputs from one gated frame."""

    events: list[CaptionEvent]
    caption_sources: list[str]
    detections: list[dict[str, Any]]
    routing: RoutingDecision | None
    active_agents: frozenset[str]
    agent_results: dict[str, Any] | None
    detect_ms: float | None
    router_ms: float | None
    manager_ms: float | None
    template_ms: float
    e2e_ms: float


class AppRuntime:
    """Holds agents and runs one frame through detect → caption."""

    def __init__(
        self,
        cfg: Any,
        *,
        dry_run: bool = False,
        debug_overlay: bool | None = None,
    ) -> None:
        self._cfg = cfg
        self._dry_run = dry_run
        self._object_agent = ObjectAgent(
            model_path=cfg.yolo_model_path,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            max_detections=cfg.max_detections,
        )
        self._engine = SpatialTemplateEngine(empty_policy=cfg.empty_policy)
        kb: KnowledgeBase | None = None
        if cfg.rag_enabled:
            try:
                kb = KnowledgeBase(
                    cfg.rag_model_path,
                    cfg.rag_knowledge_dir,
                    top_k=cfg.rag_top_k,
                    min_similarity=cfg.rag_min_similarity,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[app] RAG disabled (init failed): {exc}", flush=True)
                kb = None
        self._orchestrator = CaptionOrchestrator(
            template_engine=self._engine,
            knowledge_base=kb,
            rag_enabled=bool(cfg.rag_enabled and kb is not None),
            dedup_similarity=cfg.rag_dedup_similarity,
        )
        overlay_flag = cfg.debug_overlay if debug_overlay is None else debug_overlay
        self._overlay = CaptionDebugOverlay(bool(overlay_flag))
        self._audio = AudioOutputManager(
            model_path=cfg.piper_model_path,
            output_dir=cfg.reports_dir / "audio",
            dedup_window_s=cfg.dedup_window_s,
        )
        if not dry_run:
            self._audio.start()
        self._encoder = (
            VisualEncoder(cfg.mobilenet_model_path) if cfg.use_encoder else None
        )

        self._router: SLMRouter | None = None
        self._ring: Any = None
        self._manager: AgentManager | None = None
        self._action_agent: ActionAgent | None = None

        if cfg.tier1_enabled and not cfg.tier1_object_only:
            strict = cfg.tier1_strict_artifacts
            scene_path = cfg.tier1_router_path
            self._router = SLMRouter(
                scene_path,
                confidence_threshold=cfg.tier1_router_confidence,
                image_size=cfg.tier1_router_image_size,
                scene_to_agents={
                    k: list(v) for k, v in cfg.tier1_scene_to_agents.items()
                },
                fallback=cfg.tier1_fallback,
                strict_artifacts=strict,
            )
            from src.capture.ring_buffer import RingBuffer

            self._ring = RingBuffer(cfg.tier1_ring_maxlen)
            self._action_agent = ActionAgent(
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
            self._manager = AgentManager(
                object_agent=self._object_agent,
                action_agent=self._action_agent,
                ocr_agent=ocr_agent,
                face_pose_agent=face_pose_agent,
                timeout_object_s=cfg.tier1_timeouts["object"],
                timeout_action_s=cfg.tier1_timeouts["action"],
                timeout_ocr_s=cfg.tier1_timeouts["ocr"],
                timeout_face_pose_s=cfg.tier1_timeouts["face_pose"],
            )

        self._prev_processed_rgb: np.ndarray | None = None

    def set_rag_enabled(self, enabled: bool) -> None:
        """Toggle RAG retrieval (e.g. demo hotkey)."""

        self._orchestrator.set_rag_enabled(enabled)

    @property
    def orchestrator(self) -> CaptionOrchestrator:
        return self._orchestrator

    @property
    def object_agent(self) -> ObjectAgent:
        return self._object_agent

    @property
    def audio(self) -> AudioOutputManager:
        return self._audio

    @property
    def overlay(self) -> CaptionDebugOverlay:
        return self._overlay

    @property
    def router(self) -> SLMRouter | None:
        return self._router

    @property
    def manager(self) -> AgentManager | None:
        return self._manager

    @property
    def ring(self) -> Any:
        return self._ring

    @property
    def encoder(self) -> VisualEncoder | None:
        return self._encoder

    def close(self) -> None:
        if not self._dry_run:
            self._audio.stop()
        self._overlay.close()
        if self._manager is not None:
            self._manager.close()

    def process_frame(
        self,
        packet: FramePacket,
        *,
        tick_start: float,
    ) -> FrameProcessResult:
        """Run full pipeline for one frame (after change gate)."""

        if self._encoder is not None:
            _ = self._encoder.encode_packet(packet)

        cfg = self._cfg
        routing: RoutingDecision | None = None
        active: frozenset[str] = frozenset({"object"})
        router_ms: float | None = None
        manager_ms: float | None = None
        detect_ms: float | None = None

        if cfg.tier1_enabled and not cfg.tier1_object_only and self._router is not None:
            t_r0 = time.perf_counter()
            routing = self._router.route(packet.data, self._prev_processed_rgb)
            router_ms = (time.perf_counter() - t_r0) * 1000.0
            active = routing.active_agents
            self._prev_processed_rgb = packet.data.copy()
        elif cfg.tier1_enabled and cfg.tier1_object_only:
            active = frozenset({"object"})

        if self._ring is not None:
            self._ring.push(packet)

        action_packets = None
        if (
            self._ring is not None
            and "action" in active
            and self._action_agent is not None
            and self._manager is not None
        ):
            packets_list = self._ring.as_list()
            if len(packets_list) >= 1:
                action_packets = packets_list[-self._action_agent.input_frames :]

        agent_results: dict[str, Any] | None = None
        tier1_mgr = cfg.tier1_enabled and not cfg.tier1_object_only
        tier1_mgr = tier1_mgr and self._manager is not None
        if tier1_mgr:
            mgr = self._manager
            assert mgr is not None
            t_mgr0 = time.perf_counter()
            agent_results = mgr.run_parallel(
                frame_rgb=packet.data,
                active=active,
                action_packets=action_packets,
            )
            dets = agent_results.get("object")
            if dets is not None and dets.status == "ok":
                detections = dets.payload
            elif "object" in active:
                detections = self._object_agent.detect_instances(packet.data)
            else:
                detections = []
            manager_ms = (time.perf_counter() - t_mgr0) * 1000.0

            t_cap = time.perf_counter()
            events = self._orchestrator.generate(
                detections,
                frame_width=packet.data.shape[1],
                frame_height=packet.data.shape[0],
                scene_type=routing.scene_type if routing is not None else None,
                agent_results=agent_results,
                merge_tier1=True,
            )
            template_ms = (time.perf_counter() - t_cap) * 1000.0
        else:
            agent_results = None
            t0 = time.perf_counter()
            detections = self._object_agent.detect_instances(packet.data)
            detect_ms = (time.perf_counter() - t0) * 1000.0

            t_cap = time.perf_counter()
            events = self._orchestrator.generate(
                detections,
                frame_width=int(packet.data.shape[1]),
                frame_height=packet.data.shape[0],
                scene_type=None,
                agent_results=None,
                merge_tier1=False,
            )
            template_ms = (time.perf_counter() - t_cap) * 1000.0

        e2e_ms = (time.perf_counter() - tick_start) * 1000.0
        caption_sources = self._orchestrator.describe_sources(events)

        return FrameProcessResult(
            events=list(events),
            caption_sources=caption_sources,
            detections=detections,
            routing=routing,
            active_agents=active,
            agent_results=agent_results,
            detect_ms=detect_ms,
            router_ms=router_ms,
            manager_ms=manager_ms,
            template_ms=template_ms,
            e2e_ms=e2e_ms,
        )


def build_pipeline_status(
    *,
    fps: float,
    latency_ms: float,
    result: FrameProcessResult,
    verbosity: VerbosityLevel,
    is_muted: bool,
) -> PipelineStatus:
    srcs = result.caption_sources
    if not srcs:
        cap_src = "template"
    elif all(s == "rag" for s in srcs):
        cap_src = "rag"
    elif all(s == "template" for s in srcs):
        cap_src = "template"
    else:
        cap_src = "mixed"
    scene = result.routing.scene_type if result.routing is not None else ""
    return PipelineStatus(
        fps=fps,
        pipeline_latency_ms=latency_ms,
        active_agents=sorted(result.active_agents),
        scene_type=scene,
        detection_count=len(result.detections),
        caption_source=cap_src,
        verbosity=verbosity,
        is_muted=is_muted,
    )
