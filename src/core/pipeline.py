"""Threaded capture hook (optional); delegates to ``src.app.main``."""

from __future__ import annotations

from typing import Any

from src.app.main import AppConfig, run_app


def run_app_threaded(
    cfg: AppConfig,
    *,
    max_frames: int | None = None,
    source_override: Any | None = None,
    dry_run: bool = False,
    visual: bool = False,
    enable_tts: bool = True,
    verbosity: str = "medium",
    hazard_beep: bool = True,
) -> dict[str, Any]:
    """Same as ``run_app(..., use_threaded_pipeline=True)``."""

    return run_app(
        cfg,
        max_frames=max_frames,
        source_override=source_override,
        dry_run=dry_run,
        visual=visual,
        enable_tts=enable_tts,
        use_threaded_pipeline=True,
        verbosity=verbosity,
        hazard_beep=hazard_beep,
    )
