"""Tiện ích augmentation ảnh cho pipeline thị giác.

Module tạo preset Albumentations cho detection, scene và OCR. Preview mode chỉ
đếm candidate images; execute mode mới ghi ảnh output xuống đĩa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.training.data.core import (
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    IMAGE_EXTENSIONS,
)


@dataclass(frozen=True)
class AugmentationPreset:
    """Metadata preset augmentation cho một image task."""

    task: str
    size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    supports_bboxes: bool


def augmentation_preset(task: str, size: int = 640) -> AugmentationPreset:
    """Trả về metadata preset augmentation theo task.

    Task hợp lệ là `detection`, `scene`, hoặc `ocr`; riêng detection bật bbox
    support theo format YOLO.
    """

    normalized_task = task.lower()
    if normalized_task not in {"detection", "scene", "ocr"}:
        raise ValueError("Expected task to be one of: detection, scene, ocr")
    if size <= 0:
        raise ValueError("size must be positive")
    return AugmentationPreset(
        task=normalized_task,
        size=size,
        normalize_mean=DEFAULT_NORMALIZE_MEAN,
        normalize_std=DEFAULT_NORMALIZE_STD,
        supports_bboxes=normalized_task == "detection",
    )


def build_image_augmentation(
    task: str,
    size: int = 640,
    normalize: bool = True,
) -> Any:
    """Tạo Albumentations transform cho detection, scene hoặc OCR."""

    import albumentations

    preset = augmentation_preset(task, size)
    transforms: list[Any]
    if preset.task == "ocr":
        transforms = [
            albumentations.Resize(height=size, width=size),
            albumentations.RandomBrightnessContrast(p=0.4),
        ]
    else:
        transforms = [
            albumentations.LongestMaxSize(max_size=size),
            albumentations.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
            albumentations.RandomCrop(height=size, width=size, p=0.25),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.3),
            albumentations.RandomBrightnessContrast(p=0.3),
        ]
    if normalize:
        transforms.append(
            albumentations.Normalize(
                mean=preset.normalize_mean,
                std=preset.normalize_std,
            )
        )
    if preset.supports_bboxes:
        return albumentations.Compose(
            transforms,
            bbox_params=albumentations.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )
    return albumentations.Compose(transforms)


def augment_preview(
    input_dir: Path,
    output_dir: Path,
    limit: int = 8,
    dry_run: bool = True,
    task: str = "detection",
    size: int = 640,
) -> int:
    """Preview hoặc ghi một số ảnh augmentation mẫu."""

    image_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:limit]
    if dry_run:
        return len(image_paths)

    import numpy as np

    transform = build_image_augmentation(task, size=size, normalize=False)
    preset = augmentation_preset(task, size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        if preset.supports_bboxes:
            result = transform(
                image=image_array,
                bboxes=[],
                class_labels=[],
            )
        else:
            result = transform(image=image_array)
        Image.fromarray(result["image"]).save(output_dir / image_path.name)
    return len(image_paths)
