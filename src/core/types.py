"""Shared dataclasses for pipeline status and UX (UI + logging)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VerbosityLevel = Literal["low", "medium", "high"]


@dataclass(slots=True)
class PipelineStatus:
    """Single frame snapshot for dashboard / Demo UI."""

    fps: float
    pipeline_latency_ms: float
    active_agents: list[str] = field(default_factory=list)
    scene_type: str = ""
    detection_count: int = 0
    caption_source: str = "template"  # "rag" | "template" | "mixed"
    verbosity: VerbosityLevel = "medium"
    is_muted: bool = False
