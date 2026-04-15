"""RAG-first caption orchestrator with spatial template fallback."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from src.agents.agent_manager import AgentResult
from src.caption.template_engine import (
    CaptionEvent,
    SpatialTemplateEngine,
    _direction_from_bbox,
    _distance_from_bbox,
)
from src.caption.tier1_caption import merge_tier1_caption_events
from src.rag.knowledge_base import KnowledgeBase
from src.rag.types import ScoredEntry

_VI_TO_EN_DIR = {
    "bên trái": "left",
    "bên phải": "right",
    "phía trước": "front",
}
_VI_TO_EN_DIST = {
    "gần": "close",
    "trung bình": "medium",
    "xa": "far",
}


def _priority_int_to_str(p: int) -> str:
    if p <= 0:
        return "hazard"
    if p == 1:
        return "navigation"
    return "info"


@dataclass(frozen=True, slots=True)
class CaptionResult:
    """One utterance with provenance for logging and TTS."""

    text: str
    priority: str
    source: Literal["rag", "template"]
    signature: str


def _bbox_ok(bbox: Any) -> bool:
    return isinstance(bbox, dict) and all(k in bbox for k in ("x1", "y1", "x2", "y2"))


def build_rag_query_string(
    detections: list[dict[str, Any]],
    *,
    scene_type: str | None,
    frame_width: int,
    frame_height: int,
    agent_results: dict[str, AgentResult] | None = None,
) -> str:
    """English keyword string for MiniLM retrieval."""

    parts: list[str] = []
    if scene_type:
        parts.append(str(scene_type).replace("_", " "))
    for det in detections:
        label = str(det.get("label", "")).strip()
        if not label:
            continue
        bbox = det.get("bbox", {})
        if _bbox_ok(bbox):
            d_vn = _direction_from_bbox(bbox, float(frame_width))
            t_vn = _distance_from_bbox(
                bbox, float(frame_width), float(frame_height)
            )
            d_en = _VI_TO_EN_DIR.get(d_vn, d_vn)
            t_en = _VI_TO_EN_DIST.get(t_vn, t_vn)
            parts.append(f"{label} {d_en} {t_en}")
        else:
            parts.append(label)
    ar = agent_results or {}
    act = ar.get("action")
    if act is not None and act.status == "ok" and isinstance(act.payload, dict):
        a = str(act.payload.get("action", "")).strip()
        if a:
            parts.append(f"action {a}")
    oc = ar.get("ocr")
    if oc is not None and oc.status == "ok" and isinstance(oc.payload, list):
        for line in oc.payload[:2]:
            if isinstance(line, dict):
                tx = str(line.get("text", "")).strip()
                if tx:
                    parts.append(tx[:40])
    fp = ar.get("face_pose")
    if fp is not None and fp.status == "ok" and isinstance(fp.payload, dict):
        for face in (fp.payload.get("faces") or [])[:1]:
            if isinstance(face, dict):
                em = str(face.get("emotion", "")).strip()
                if em:
                    parts.append(f"face {em}")
        for pose in (fp.payload.get("poses") or [])[:1]:
            if isinstance(pose, dict):
                pt = str(pose.get("pose_type", "")).strip()
                if pt:
                    parts.append(f"pose {pt}")
    return " ".join(parts).strip()


def _scored_to_events(hits: list[ScoredEntry]) -> list[CaptionEvent]:
    """Convert retrieval hits to caption events (hazard first)."""

    ranked = sorted(
        hits,
        key=lambda h: (h.entry.priority, -h.score),
    )
    events: list[CaptionEvent] = []
    for h in ranked:
        e = h.entry
        pr = _priority_int_to_str(e.priority)
        sig = f"rag:{e.query}:{h.score:.3f}"
        events.append(
            CaptionEvent(
                text=e.caption_vi,
                priority=pr,
                label=e.query[:48],
                direction="",
                distance="",
                confidence=float(h.score),
                signature=sig,
            )
        )
    return events


def _similar_enough(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= threshold


class CaptionOrchestrator:
    """RAG search with template fallback; optional Tier-1 merge for extra experts."""

    def __init__(
        self,
        *,
        template_engine: SpatialTemplateEngine,
        knowledge_base: KnowledgeBase | None,
        rag_enabled: bool,
        dedup_similarity: float = 0.8,
    ) -> None:
        self._engine = template_engine
        self._kb = knowledge_base
        self._rag_enabled = rag_enabled
        self._dedup_sim = float(dedup_similarity)
        self._last_text: str | None = None

    def generate(
        self,
        detections: list[dict[str, Any]],
        *,
        frame_width: int,
        frame_height: int,
        scene_type: str | None = None,
        agent_results: dict[str, AgentResult] | None = None,
        merge_tier1: bool = False,
    ) -> list[CaptionEvent]:
        """Return caption events for TTS; RAG when enabled and confident, else template."""

        agent_results = agent_results or {}
        query = build_rag_query_string(
            detections,
            scene_type=scene_type,
            frame_width=frame_width,
            frame_height=frame_height,
            agent_results=agent_results,
        )
        base_events: list[CaptionEvent]

        used_rag = False
        if (
            self._rag_enabled
            and self._kb is not None
            and query
        ):
            hits = self._kb.search(query)
            if hits:
                base_events = _scored_to_events(hits)
                used_rag = True
            else:
                base_events = self._engine.render(
                    detections,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
        else:
            base_events = self._engine.render(
                detections,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        if merge_tier1 and agent_results:
            events = merge_tier1_caption_events(base_events, agent_results)
        else:
            events = list(base_events)

        events = self._dedup_events(events)
        return events

    def _dedup_events(self, events: list[CaptionEvent]) -> list[CaptionEvent]:
        if not events:
            return events
        out: list[CaptionEvent] = []
        for ev in events:
            if self._last_text is not None and _similar_enough(
                ev.text, self._last_text, self._dedup_sim
            ):
                continue
            out.append(ev)
            self._last_text = ev.text
        return out

    def describe_sources(self, events: list[CaptionEvent]) -> list[str]:
        """Return parallel source labels for logging (rag vs template)."""

        return [
            "rag" if e.signature.startswith("rag:") else "template"
            for e in events
        ]
