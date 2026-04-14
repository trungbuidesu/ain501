# -*- coding: utf-8 -*-
"""Tiện ích YOLO dataset cho object detection.

Module validate layout ảnh/label kiểu Ultralytics, đọc class names, preview
hoặc execute remap class id khi merge custom exports với COCO.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.training.data.core import (
    IMAGE_EXTENSIONS,
    MergeReport,
    ValidationIssue,
    ValidationReport,
    load_yaml,
)


@dataclass(frozen=True)
class NormalizeYoloReport:
    """Report returned after normalizing a third-party YOLO export.

    `validation` contains all blocking issues discovered while reading,
    remapping, copying, and optionally revalidating the normalized output.
    """

    dry_run: bool
    scanned_labels: int
    copied_files: int
    validation: ValidationReport

    @property
    def valid(self) -> bool:
        """Cho biết normalization không có issue cấp `error`."""
        return self.validation.valid


def validate_yolo_dataset(root: Path, class_names: Sequence[str]) -> ValidationReport:
    """Validate layout YOLO kiểu Ultralytics.

    Kiểm tra ảnh corrupt, cặp label-image, class id và bbox range. Trả về
    `ValidationReport` mà không thay đổi dữ liệu.
    """

    issues: list[ValidationIssue] = []
    checked_files = 0
    labels_root = root / "labels"
    images_root = root / "images"

    label_paths = sorted(labels_root.rglob("*.txt")) if labels_root.exists() else []
    image_paths = [
        path
        for path in sorted(images_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_stems = {
        path.relative_to(images_root).with_suffix("") for path in image_paths
    }
    label_stems = {
        path.relative_to(labels_root).with_suffix("") for path in label_paths
    }

    for image_path in image_paths:
        checked_files += 1
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            issues.append(
                ValidationIssue("error", str(image_path), f"corrupt image: {exc}")
            )
        if image_path.relative_to(images_root).with_suffix("") not in label_stems:
            issues.append(
                ValidationIssue("error", str(image_path), "missing YOLO label file")
            )

    for label_path in label_paths:
        checked_files += 1
        if label_path.relative_to(labels_root).with_suffix("") not in image_stems:
            issues.append(
                ValidationIssue("error", str(label_path), "missing matching image")
            )
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            issue = validate_yolo_label_line(
                line,
                class_count=len(class_names),
                path=f"{label_path}:{line_number}",
            )
            if issue is not None:
                issues.append(issue)

    return ValidationReport(checked_files, tuple(issues))


def validate_yolo_label_line(
    line: str,
    class_count: int,
    path: str,
) -> ValidationIssue | None:
    """Validate một dòng YOLO label đã chuẩn hóa."""

    parts = line.split()
    if len(parts) != 5:
        return ValidationIssue("error", path, "YOLO label must have 5 fields")
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        return ValidationIssue("error", path, f"non-numeric YOLO label: {exc}")
    if class_id < 0 or class_id >= class_count:
        return ValidationIssue("error", path, f"unknown class id: {class_id}")
    cx, cy, width, height = values
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
        return ValidationIssue("error", path, "YOLO bbox values must be in [0, 1]")
    if width <= 0.0 or height <= 0.0:
        return ValidationIssue("error", path, "YOLO width/height must be positive")
    return None


def merge_yolo_datasets(
    sources: Sequence[Path],
    output_root: Path,
    canonical_names: Sequence[str],
    dry_run: bool = True,
) -> MergeReport:
    """Merge nhiều dataset YOLO theo canonical class names.

    Dry-run chỉ quét labels và cảnh báo remap. Execute mode copy ảnh/label vào
    `output_root` và ghi `classes.txt` chuẩn.
    """

    issues: list[ValidationIssue] = []
    scanned_labels = 0
    copied_files = 0
    remaps: dict[str, dict[int, int]] = {}
    canonical_index = {name: index for index, name in enumerate(canonical_names)}

    for source in sources:
        source_names = read_yolo_names(source)
        source_remap: dict[int, int] = {}
        for source_id, name in enumerate(source_names):
            if name in canonical_index:
                source_remap[source_id] = canonical_index[name]
            else:
                issues.append(
                    ValidationIssue(
                        "error",
                        str(source),
                        f"source class '{name}' not in canonical names",
                    )
                )
        remaps[str(source)] = source_remap

        for label_path in sorted((source / "labels").rglob("*.txt")):
            scanned_labels += 1
            relative_label = label_path.relative_to(source / "labels")
            relative_image = _find_matching_image(source / "images", relative_label)
            if relative_image is None:
                issues.append(
                    ValidationIssue("error", str(label_path), "missing matching image")
                )
                continue
            remapped_lines = remap_yolo_label_text(
                label_path.read_text(encoding="utf-8"),
                source_remap,
                str(label_path),
                issues,
            )
            if dry_run:
                continue
            output_label = output_root / "labels" / relative_label
            output_image = output_root / "images" / relative_image
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(remapped_lines, encoding="utf-8")
            shutil.copy2(source / "images" / relative_image, output_image)
            copied_files += 2

    if not dry_run:
        (output_root / "classes.txt").write_text(
            "\n".join(canonical_names) + "\n",
            encoding="utf-8",
        )
    return MergeReport(dry_run, scanned_labels, copied_files, remaps, tuple(issues))


def read_yolo_names(source: Path) -> list[str]:
    """Đọc YOLO class names từ `classes.txt`, `.names` hoặc `data.yaml`."""

    for classes_path in (source / "classes.txt", source / "obj.names"):
        if classes_path.exists():
            return _read_yolo_names_from_file(classes_path)
    for classes_path in source.glob("*.names"):
        if classes_path.exists():
            return _read_yolo_names_from_file(classes_path)
    data_yaml_path = source / "data.yaml"
    if data_yaml_path.exists():
        return _read_yolo_names_from_file(data_yaml_path)
    raise ValueError(f"Cannot find YOLO class names in {source}")


def _read_yolo_names_from_file(classes_path: Path) -> list[str]:
    """Đọc class names từ YOLO metadata file.

    Text files được đọc theo từng dòng; YAML files dùng field `names`.
    """

    if classes_path.suffix.lower() in {".yaml", ".yml"}:
        data = load_yaml(classes_path)
        names = data.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        raise ValueError(f"Cannot find names in {classes_path}")
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_custom_yolo_export(
    export_root: Path,
    output_root: Path,
    canonical_names: Sequence[str],
    source_format: str,
    classes_path: Path | None = None,
    dry_run: bool = True,
) -> NormalizeYoloReport:
    """Normalize export YOLO từ CVAT, Roboflow hoặc Ultralytics.

    Source class IDs được remap về canonical class order. Execute mode copy
    ảnh/label sang layout `images/<split>` + `labels/<split>` và validate lại.
    """

    if source_format not in {"roboflow_yolo", "cvat_yolo", "ultralytics_yolo"}:
        raise ValueError("Unsupported YOLO source format")
    if not export_root.exists():
        validation = ValidationReport(
            0,
            (ValidationIssue("error", str(export_root), "missing export root"),),
        )
        return NormalizeYoloReport(dry_run, 0, 0, validation)

    source_names = (
        _read_yolo_names_from_file(classes_path)
        if classes_path is not None
        else read_yolo_names(export_root)
    )
    canonical_index = {name: index for index, name in enumerate(canonical_names)}
    class_remap: dict[int, int] = {}
    issues: list[ValidationIssue] = []
    for source_id, name in enumerate(source_names):
        if name in canonical_index:
            class_remap[source_id] = canonical_index[name]
        else:
            issues.append(
                ValidationIssue(
                    "error",
                    str(export_root),
                    f"source class '{name}' not in canonical names",
                )
            )

    scanned_labels = 0
    copied_files = 0
    label_paths = _custom_yolo_label_paths(export_root)
    if not label_paths:
        issues.append(
            ValidationIssue("error", str(export_root), "no YOLO label files found")
        )
    for label_path in label_paths:
        scanned_labels += 1
        split = _infer_yolo_split(label_path.relative_to(export_root))
        image_path = _find_custom_yolo_image(export_root, label_path)
        if image_path is None:
            issues.append(
                ValidationIssue("error", str(label_path), "missing matching image")
            )
            continue
        remapped = remap_yolo_label_text(
            label_path.read_text(encoding="utf-8"),
            class_remap,
            str(label_path),
            issues,
        )
        if dry_run:
            continue
        output_label = output_root / "labels" / split / label_path.name
        output_image = output_root / "images" / split / image_path.name
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text(remapped, encoding="utf-8")
        shutil.copy2(image_path, output_image)
        copied_files += 2

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "classes.txt").write_text(
            "\n".join(canonical_names) + "\n",
            encoding="utf-8",
        )
        issues.extend(validate_yolo_dataset(output_root, canonical_names).issues)
    validation = ValidationReport(scanned_labels, tuple(issues))
    return NormalizeYoloReport(dry_run, scanned_labels, copied_files, validation)


def _custom_yolo_label_paths(export_root: Path) -> list[Path]:
    """Liệt kê label files và bỏ qua metadata text của YOLO export."""

    metadata_names = {
        "classes.txt",
        "obj.names",
        "test.txt",
        "train.txt",
        "valid.txt",
        "val.txt",
    }
    cvat_dirs = {"obj_train_data", "obj_valid_data", "obj_test_data"}
    label_paths: list[Path] = []
    for path in sorted(export_root.rglob("*.txt")):
        if path.name in metadata_names:
            continue
        lowercase_parts = {part.lower() for part in path.parts}
        is_images_labels_layout = "labels" in lowercase_parts
        is_cvat_flat_layout = path.parent.name.lower() in cvat_dirs
        if is_images_labels_layout or is_cvat_flat_layout:
            label_paths.append(path)
    return label_paths


def _infer_yolo_split(relative_path: Path) -> str:
    """Suy diễn output split từ đường dẫn export YOLO/CVAT."""

    for part in relative_path.parts:
        normalized = part.lower()
        if normalized == "valid":
            return "val"
        if normalized in {"train", "val", "test"}:
            return normalized
        if normalized == "obj_valid_data":
            return "val"
        if normalized == "obj_test_data":
            return "test"
        if normalized == "obj_train_data":
            return "train"
    return "train"


def _find_custom_yolo_image(export_root: Path, label_path: Path) -> Path | None:
    """Tìm ảnh tương ứng với YOLO label trong export root."""

    stem = label_path.with_suffix("")
    for extension in IMAGE_EXTENSIONS:
        flat_candidate = stem.with_suffix(extension)
        if flat_candidate.exists():
            return flat_candidate
    parts = list(label_path.relative_to(export_root).parts)
    lowercase_parts = [part.lower() for part in parts]
    if "labels" in lowercase_parts:
        parts[lowercase_parts.index("labels")] = "images"
        candidate_relative = Path(*parts).with_suffix("")
        for extension in IMAGE_EXTENSIONS:
            candidate = export_root / candidate_relative.with_suffix(extension)
            if candidate.exists():
                return candidate
    for extension in IMAGE_EXTENSIONS:
        matches = sorted(export_root.rglob(f"{label_path.stem}{extension}"))
        if matches:
            return matches[0]
    return None


def _find_matching_image(images_root: Path, relative_label: Path) -> Path | None:
    """Tìm ảnh tương ứng từ label stem và các extension hỗ trợ."""
    stem = relative_label.with_suffix("")
    for extension in IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(extension)
        if (images_root / candidate).exists():
            return candidate
    return None


def remap_yolo_label_text(
    label_text: str,
    class_remap: dict[int, int],
    path: str,
    issues: list[ValidationIssue],
) -> str:
    """Remap label text và ghi issue khi class id không hợp lệ."""

    output_lines: list[str] = []
    for line_number, line in enumerate(label_text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            source_id = int(parts[0])
        except ValueError:
            issues.append(
                ValidationIssue("error", f"{path}:{line_number}", "invalid class id")
            )
            continue
        if source_id not in class_remap:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}:{line_number}",
                    f"class id {source_id} has no remap",
                )
            )
            continue
        output_lines.append(" ".join([str(class_remap[source_id]), *parts[1:]]))
    return "\n".join(output_lines) + ("\n" if output_lines else "")


def canonical_class_names(
    config: dict[str, Any],
    class_key: str = "coco_subset",
    include_custom: bool = False,
) -> list[str]:
    """Đọc canonical classes từ dataset config."""

    classes = config.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("Config missing classes mapping")
    names = classes.get(class_key, [])
    if not isinstance(names, list):
        raise ValueError(f"Config classes.{class_key} must be a list")
    class_names = [str(name) for name in names]
    custom_names = classes.get("custom_only", [])
    if include_custom and isinstance(custom_names, list):
        class_names.extend(str(name) for name in custom_names)
    return class_names


def write_classes_file(
    config_path: Path,
    output_path: Path,
    class_key: str = "coco_subset",
    include_custom: bool = False,
    dry_run: bool = True,
) -> list[str]:
    """Preview hoặc ghi `classes.txt` từ dataset config."""

    class_names = canonical_class_names(
        load_yaml(config_path),
        class_key=class_key,
        include_custom=include_custom,
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return class_names
