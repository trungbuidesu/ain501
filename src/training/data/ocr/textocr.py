# -*- coding: utf-8 -*-
"""OCR scene-text dataset helpers.

TextOCR is the current Phase 0 target and ICDAR parsing is kept only as a
legacy fixture/helper. TextOCR annotations are normalized into text boxes
with image id, optional image filename, bbox, polygon points, text, and an
ignored flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IcdarTextBox:
    """Một annotation text localization theo format ICDAR."""

    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str
    ignored: bool


@dataclass(frozen=True)
class TextOcrTextBox:
    """Một annotation text localization theo format TextOCR."""

    image_id: str
    image_file: str | None
    bbox: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    text: str
    ignored: bool


def parse_icdar_gt(path: Path) -> list[IcdarTextBox]:
    """Parse legacy ICDAR 2015 ground-truth text boxes.

    Kept for backwards-compatible fixtures and possible old exports. The
    current OCR target is TextOCR, so new pipeline code should prefer
    `parse_textocr_annotations`.
    """

    entries: list[IcdarTextBox] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split(",", maxsplit=8)
        if len(parts) != 9:
            raise ValueError(f"Invalid ICDAR line in {path}: {line}")
        coords = [int(value) for value in parts[:8]]
        points = (
            (coords[0], coords[1]),
            (coords[2], coords[3]),
            (coords[4], coords[5]),
            (coords[6], coords[7]),
        )
        text = parts[8]
        entries.append(IcdarTextBox(points, text, ignored=text == "###"))
    return entries


def parse_textocr_annotations(path: Path) -> list[TextOcrTextBox]:
    """Normalize a TextOCR JSON annotation file into text boxes.

    Supports the common `anns`/`imgs` mapping shape and a list-style
    `annotations` fallback. Entries without bbox or polygon points are skipped;
    empty or illegible text is marked ignored.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = _textocr_annotation_values(data)
    images_by_id = _textocr_images_by_id(data.get("imgs", {}))
    entries: list[TextOcrTextBox] = []

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        image_id = str(annotation.get("image_id", annotation.get("imageId", "")))
        text = str(
            annotation.get(
                "utf8_string",
                annotation.get("text", annotation.get("transcription", "")),
            )
        )
        bbox = _textocr_bbox(annotation.get("bbox"))
        points = _textocr_points(annotation.get("points"))
        if bbox is None or points is None:
            continue
        image_meta = images_by_id.get(image_id, {})
        image_file = image_meta.get("file_name")
        legibility = str(annotation.get("legibility", "")).lower()
        entries.append(
            TextOcrTextBox(
                image_id=image_id,
                image_file=str(image_file) if image_file is not None else None,
                bbox=bbox,
                points=points,
                text=text,
                ignored=(not text.strip()) or legibility == "illegible",
            )
        )
    return entries


def _textocr_annotation_values(data: Any) -> list[Any]:
    """Trả về danh sách annotation từ TextOCR JSON dạng dict hoặc list."""

    if not isinstance(data, dict):
        raise ValueError("Expected TextOCR JSON mapping")
    annotations = data.get("anns", data.get("annotations", []))
    if isinstance(annotations, dict):
        return list(annotations.values())
    if isinstance(annotations, list):
        return annotations
    raise ValueError("Expected TextOCR annotations in `anns` or `annotations`")


def _textocr_images_by_id(images: Any) -> dict[str, dict[str, Any]]:
    """Chuẩn hóa metadata ảnh TextOCR thành mapping theo image id."""

    if isinstance(images, dict):
        return {
            str(image_id): image
            for image_id, image in images.items()
            if isinstance(image, dict)
        }
    if isinstance(images, list):
        return {
            str(image["id"]): image
            for image in images
            if isinstance(image, dict) and image.get("id") is not None
        }
    return {}


def _textocr_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    """Chuẩn hóa bbox TextOCR `[x, y, width, height]`."""

    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return None
    return (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )


def _textocr_points(raw_points: Any) -> tuple[tuple[float, float], ...] | None:
    """Chuẩn hóa polygon TextOCR thành tuple point `(x, y)`."""

    if not isinstance(raw_points, list | tuple):
        return None
    if raw_points and all(isinstance(point, list | tuple) for point in raw_points):
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in raw_points
            if isinstance(point, list | tuple) and len(point) >= 2
        )
        return points or None
    if len(raw_points) % 2 != 0:
        return None
    return tuple(
        (float(raw_points[index]), float(raw_points[index + 1]))
        for index in range(0, len(raw_points), 2)
    )
