"""Shared UI control state (no OpenCV import)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DemoControlState:
    """Mutable UI state shared with the pipeline loop (same thread)."""

    muted: bool = False
    verbosity: str = "medium"
    rag_enabled: bool = True
    force_describe: bool = False
    read_text_request: bool = False
    screenshot_dir: Path = field(default_factory=lambda: Path("screenshots"))
