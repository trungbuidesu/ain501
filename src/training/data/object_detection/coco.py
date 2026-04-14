# -*- coding: utf-8 -*-
"""Tiện ích object detection cho COCO 2017.

Mô đun này đảm nhận logic chuyển dịch chú giải JSON (JSON Annotation) của COCO
sang định dạng Nhãn YOLO, phục vụ Giai đoạn 0 (Phase 0). Dữ liệu đầu vào yêu cầu
bản đồ ánh xạ COCO đã tải trong bộ nhớ; kết quả trả ra là chuỗi tọa độ YOLO đã
chuẩn hóa.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def normalize_coco_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Chuyển bbox COCO `[x, y, width, height]` sang YOLO.

    Trả về `(center_x, center_y, width, height)` chuẩn hóa trong `[0, 1]`.
    Bounding box được clip theo biên ảnh; bbox rỗng hoặc malformed trả về
    `None`.
    """

    if len(bbox) != 4:
        return None
    x, y, width, height = (float(value) for value in bbox)
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    cx = (x1 + clipped_width / 2.0) / image_width
    cy = (y1 + clipped_height / 2.0) / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height
    return (cx, cy, normalized_width, normalized_height)


def coco_annotations_to_yolo_lines(
    coco: dict[str, Any],
    class_names: Sequence[str],
) -> dict[str, list[str]]:
    """Tạo YOLO label lines cho các COCO classes đã cấu hình.

    Chỉ giữ annotations thuộc `class_names`. Mapping trả về được key bằng
    COCO file name để `write_yolo_from_coco` có thể ghi thẳng ra disk.
    """

    images = {
        int(image["id"]): image
        for image in coco.get("images", [])
        if isinstance(image, dict) and "id" in image
    }
    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
        if isinstance(category, dict) and "id" in category and "name" in category
    }
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels_by_file: dict[str, list[str]] = {}

    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image = images.get(int(annotation.get("image_id", -1)))
        if image is None:
            continue
        category_name = category_id_to_name.get(int(annotation.get("category_id", -1)))
        if category_name not in class_to_index:
            continue
        normalized = normalize_coco_bbox(
            annotation.get("bbox", []),
            int(image.get("width", 0)),
            int(image.get("height", 0)),
        )
        if normalized is None:
            continue
        class_index = class_to_index[category_name]
        bbox_values = " ".join(f"{value:.6f}" for value in normalized)
        file_name = str(image["file_name"])
        labels_by_file.setdefault(file_name, []).append(f"{class_index} {bbox_values}")
    return labels_by_file


def write_yolo_from_coco(
    annotation_path: Path,
    image_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    split: str,
    dry_run: bool = True,
) -> int:
    """Convert một COCO annotation file thành YOLO split.

    `dry_run=True` chỉ trả số ảnh được chọn. Execute mode copy ảnh khớp,
    ghi labels và cập nhật `classes.txt` dưới `output_root`.
    """

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    labels_by_file = coco_annotations_to_yolo_lines(coco, class_names)
    if dry_run:
        return len(labels_by_file)

    images_out = output_root / "images" / split
    labels_out = output_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for file_name, lines in labels_by_file.items():
        source_image = image_root / file_name
        if source_image.exists():
            shutil.copy2(source_image, images_out / Path(file_name).name)
        (labels_out / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    (output_root / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    return len(labels_by_file)
