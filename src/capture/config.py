"""Load :file:`configs/capture_pipeline.yaml` into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.capture.types import ScreenRegion


@dataclass(frozen=True, slots=True)
class SourceConfig:
    type: Literal["webcam", "screen", "file"]
    webcam_index: int
    file_path: Path | None
    file_loop: bool


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    backend: Literal["auto", "dxcam", "mss"]
    region: ScreenRegion | None
    select_region_on_start: bool


@dataclass(frozen=True, slots=True)
class TimingConfig:
    target_fps: float
    buffer_size: int


@dataclass(frozen=True, slots=True)
class ChangeDetectionConfig:
    method: Literal["ssim", "pixel_diff"]
    ssim_trigger_below: float
    changed_fraction_trigger_above: float
    pixel_diff_threshold: int
    max_metric_side: int | None


@dataclass(frozen=True, slots=True)
class CapturePipelineConfig:
    source: SourceConfig
    screen: ScreenConfig
    timing: TimingConfig
    change_detection: ChangeDetectionConfig


def _region_from_raw(raw: Any) -> ScreenRegion | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("screen.region must be a mapping or null")
    return ScreenRegion(
        left=int(raw["left"]),
        top=int(raw["top"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
    )


def load_capture_config(path: Path | str) -> CapturePipelineConfig:
    """Parse YAML into :class:`CapturePipelineConfig`."""

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    s = raw["source"]
    src = SourceConfig(
        type=s["type"],
        webcam_index=int(s.get("webcam_index", 0)),
        file_path=Path(s["file_path"]).resolve() if s.get("file_path") else None,
        file_loop=bool(s.get("file_loop", False)),
    )

    sc = raw["screen"]
    screen = ScreenConfig(
        backend=sc.get("backend", "auto"),
        region=_region_from_raw(sc.get("region")),
        select_region_on_start=bool(sc.get("select_region_on_start", False)),
    )

    t = raw["timing"]
    timing = TimingConfig(
        target_fps=float(t["target_fps"]),
        buffer_size=int(t["buffer_size"]),
    )

    cd = raw["change_detection"]
    _ms = cd.get("max_metric_side")
    change = ChangeDetectionConfig(
        method=cd["method"],
        ssim_trigger_below=float(cd["ssim_trigger_below"]),
        changed_fraction_trigger_above=float(cd["changed_fraction_trigger_above"]),
        pixel_diff_threshold=int(cd["pixel_diff_threshold"]),
        max_metric_side=int(_ms) if _ms is not None else None,
    )

    return CapturePipelineConfig(
        source=src,
        screen=screen,
        timing=timing,
        change_detection=change,
    )
