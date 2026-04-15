"""Merge Tier 1 agent outputs into :class:`CaptionEvent` streams."""

from __future__ import annotations

from src.agents.agent_manager import AgentResult
from src.caption.template_engine import CaptionEvent


def merge_tier1_caption_events(
    object_events: list[CaptionEvent],
    agent_results: dict[str, AgentResult],
) -> list[CaptionEvent]:
    """Append non-object expert captions after spatial object captions."""

    extra: list[CaptionEvent] = []
    ar = agent_results.get("action")
    if ar is not None and ar.status == "ok" and isinstance(ar.payload, dict):
        act = str(ar.payload.get("action", ""))
        conf = float(ar.payload.get("confidence", 0.0))
        extra.append(
            CaptionEvent(
                text=f"Hành động: {act}",
                priority="info",
                label=f"action:{act}",
                direction="",
                distance="",
                confidence=conf,
                signature=f"action:{act}:{conf:.3f}",
            )
        )
    oc = agent_results.get("ocr")
    if oc is not None and oc.status == "ok" and isinstance(oc.payload, list):
        for line in oc.payload:
            if not isinstance(line, dict):
                continue
            txt = str(line.get("text", "")).strip()
            if not txt:
                continue
            cf = float(line.get("confidence", 0.0))
            extra.append(
                CaptionEvent(
                    text=f"Chữ: {txt}",
                    priority="info",
                    label="ocr",
                    direction="",
                    distance="",
                    confidence=cf,
                    signature=f"ocr:{txt}:{cf:.3f}",
                )
            )
    fp = agent_results.get("face_pose")
    if fp is not None and fp.status == "ok" and isinstance(fp.payload, dict):
        faces = fp.payload.get("faces") or []
        poses = fp.payload.get("poses") or []
        if isinstance(faces, list) and faces:
            f0 = faces[0] if isinstance(faces[0], dict) else {}
            emo = str(f0.get("emotion", "neutral"))
            sc = float(f0.get("score", 0.0))
            extra.append(
                CaptionEvent(
                    text=f"Khuôn mặt: {emo}",
                    priority="info",
                    label="face",
                    direction="",
                    distance="",
                    confidence=sc,
                    signature=f"face:{emo}",
                )
            )
        if isinstance(poses, list) and poses:
            p0 = poses[0] if isinstance(poses[0], dict) else {}
            pt = str(p0.get("pose_type", ""))
            sc2 = float(p0.get("score", 0.0))
            extra.append(
                CaptionEvent(
                    text=f"Tư thế: {pt}",
                    priority="info",
                    label="pose",
                    direction="",
                    distance="",
                    confidence=sc2,
                    signature=f"pose:{pt}",
                )
            )
    return object_events + extra
