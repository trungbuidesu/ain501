"""Spatial template engine: direction, distance, hazard-first sorting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CaptionEvent:
    text: str
    priority: str
    label: str
    direction: str
    distance: str
    confidence: float
    signature: str


def _direction_from_bbox(
    bbox: dict[str, float],
    frame_width: float,
    *,
    left_cut: float = 0.33,
    right_cut: float = 0.66,
) -> str:
    cx = (bbox["x1"] + bbox["x2"]) * 0.5
    ratio = cx / max(1.0, frame_width)
    if ratio < left_cut:
        return "bên trái"
    if ratio > right_cut:
        return "bên phải"
    return "phía trước"


def _distance_from_bbox(
    bbox: dict[str, float],
    frame_width: float,
    frame_height: float,
    *,
    near_thr: float = 0.15,
    medium_thr: float = 0.05,
) -> str:
    bw = max(1.0, bbox["x2"] - bbox["x1"])
    bh = max(1.0, bbox["y2"] - bbox["y1"])
    area_ratio = (bw * bh) / max(1.0, frame_width * frame_height)
    if area_ratio >= near_thr:
        return "gần"
    if area_ratio >= medium_thr:
        return "trung bình"
    return "xa"


class SpatialTemplateEngine:
    """Render object detections into compact navigation-friendly utterances."""

    def __init__(
        self,
        *,
        hazard_labels: set[str] | None = None,
        empty_policy: str = "silent",
    ) -> None:
        self.hazard_labels = hazard_labels or {
            "stairs",
            "car",
            "truck",
            "bus",
            "motorcycle",
        }
        self.empty_policy = empty_policy

    def _priority_for_label(self, label: str) -> str:
        return "hazard" if label in self.hazard_labels else "info"

    def render(
        self,
        detections: list[dict[str, Any]],
        *,
        frame_width: int,
        frame_height: int,
    ) -> list[CaptionEvent]:
        if not detections:
            if self.empty_policy == "message":
                return [
                    CaptionEvent(
                        text="Không có gì đặc biệt",
                        priority="info",
                        label="none",
                        direction="",
                        distance="",
                        confidence=1.0,
                        signature="none",
                    )
                ]
            return []

        scored: list[tuple[int, float, CaptionEvent]] = []
        for det in detections:
            label = str(det.get("label", "unknown"))
            conf = float(det.get("confidence", 0.0))
            bbox = det.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            direction = _direction_from_bbox(bbox, float(frame_width))
            distance = _distance_from_bbox(
                bbox,
                float(frame_width),
                float(frame_height),
            )
            priority = self._priority_for_label(label)
            text = f"{direction} {distance} có {label}"
            sig = f"{priority}:{label}:{direction}:{distance}"
            event = CaptionEvent(
                text=text,
                priority=priority,
                label=label,
                direction=direction,
                distance=distance,
                confidence=conf,
                signature=sig,
            )
            prio_rank = 0 if priority == "hazard" else 1
            scored.append((prio_rank, -conf, event))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [item[2] for item in scored]
