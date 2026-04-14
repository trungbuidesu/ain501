"""Tiện ích OCR scene text cho TextOCR và fixture ICDAR legacy.

TextOCR là dataset chính của Phase 0; parser ICDAR được giữ cho fixture cũ.
Metadata TextOCR được flatten thành text boxes chứa image id, bbox, polygon
points, text gốc và ignored flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IcdarTextBox:
    """Text localization box theo chuẩn ICDAR legacy."""

    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str
    ignored: bool


@dataclass(frozen=True)
class TextOcrTextBox:
    """Text localization box theo schema TextOCR."""

    image_id: str
    image_file: str | None
    bbox: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    text: str
    ignored: bool


def parse_icdar_gt(path: Path) -> list[IcdarTextBox]:
    """Parse ground-truth text boxes theo chuẩn ICDAR 2015.

    Hàm này được giữ để tương thích ngược; pipeline hiện tại nên ưu tiên
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
    """Parse TextOCR JSON annotation file thành text boxes.

    Hỗ trợ cả mapping `anns`/`imgs` và list `annotations`. Annotation thiếu
    bbox hoặc polygon points sẽ bị bỏ qua; text rỗng/illegible được mark ignored.
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
    """Trích xuất annotations list từ JSON TextOCR."""

    if not isinstance(data, dict):
        raise ValueError("Expected TextOCR JSON mapping")
    annotations = data.get("anns", data.get("annotations", []))
    if isinstance(annotations, dict):
        return list(annotations.values())
    if isinstance(annotations, list):
        return annotations
    raise ValueError("Expected TextOCR annotations in `anns` or `annotations`")


def _textocr_images_by_id(images: Any) -> dict[str, dict[str, Any]]:
    """Map image metadata theo image id."""

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
    """Chuẩn hóa bbox thô từ TextOCR."""

    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return None
    return (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )


def _textocr_points(raw_points: Any) -> tuple[tuple[float, float], ...] | None:
    """Chuẩn hóa polygon points thành tuple `(x, y)`."""

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
