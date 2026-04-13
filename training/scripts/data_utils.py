# -*- coding: utf-8 -*-
"""Tiện ích chuẩn bị dataset cho Phase 0 của accessibility assistant."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
DEFAULT_DATASET_CONFIG_DIR = Path("configs/datasets")
DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ValidationIssue:
    """Một lỗi hoặc cảnh báo khi validate dataset."""

    level: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Báo cáo tổng hợp trả về từ các hàm kiểm tra dữ liệu."""

    checked_files: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """Cho biết báo cáo không có lỗi mức `error`."""
        return not any(issue.level == "error" for issue in self.issues)


@dataclass(frozen=True)
class MergeReport:
    """Báo cáo trả về khi preview hoặc thực thi merge YOLO dataset."""

    dry_run: bool
    scanned_labels: int
    copied_files: int
    class_remap: dict[str, dict[int, int]]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """Cho biết quá trình merge không có lỗi mức `error`."""
        return not any(issue.level == "error" for issue in self.issues)


@dataclass(frozen=True)
class IcdarTextBox:
    """Một annotation text localization theo format ICDAR."""

    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str
    ignored: bool


@dataclass(frozen=True)
class AugmentationPreset:
    """Mô tả preset augmentation dùng cho một image task."""

    task: str
    size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    supports_bboxes: bool


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc file YAML từ đĩa và trả về mapping."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def dataset_config_paths(config_dir: Path = DEFAULT_DATASET_CONFIG_DIR) -> list[Path]:
    """Trả về danh sách file config dataset theo thứ tự ổn định."""

    return sorted(config_dir.glob("*.yaml"))


def download_plan_lines(config_paths: Sequence[Path]) -> list[str]:
    """Tạo nội dung download plan dễ đọc từ các config dataset."""

    lines: list[str] = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        lines.append(f"# {config.get('name', config_path.stem)}")
        _append_download_details(config, lines)
        lines.append("")
    return lines


def _append_download_details(node: Any, lines: list[str]) -> None:
    """Duyệt một node config và thêm thông tin download vào output."""
    if isinstance(node, dict):
        if "homepage" in node:
            lines.append(f"homepage: {node['homepage']}")
        download = node.get("download")
        if isinstance(download, dict):
            if download.get("dry_run_default") is not None:
                lines.append(f"dry_run_default: {download['dry_run_default']}")
            if download.get("manual_note"):
                lines.append(f"manual_note: {download['manual_note']}")
            files = download.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, dict):
                        lines.append(
                            f"file: {item.get('url', '<missing-url>')} -> "
                            f"{item.get('path', '<missing-path>')}"
                        )
        for value in node.values():
            _append_download_details(value, lines)
    elif isinstance(node, list):
        for value in node:
            _append_download_details(value, lines)


def normalize_coco_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Chuyển bbox COCO pixel `x, y, width, height` sang YOLO normalized."""

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
    """Chuyển annotation COCO đã chọn thành các dòng label YOLO theo file ảnh."""

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
    """Chuyển một annotation JSON COCO thành thư mục YOLO cho một split."""

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


def validate_yolo_dataset(root: Path, class_names: Sequence[str]) -> ValidationReport:
    """Kiểm tra YOLO image/label dataset tại thư mục gốc `root`."""

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
    """Kiểm tra một dòng label YOLO normalized."""

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
    """Merge nhiều YOLO dataset và remap class id của từng source."""

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
    """Đọc tên class YOLO từ `classes.txt` hoặc `data.yaml` của Ultralytics."""

    classes_path = source / "classes.txt"
    if classes_path.exists():
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    data_yaml_path = source / "data.yaml"
    if data_yaml_path.exists():
        data = load_yaml(data_yaml_path)
        names = data.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
    raise ValueError(f"Cannot find YOLO class names in {source}")


def _find_matching_image(images_root: Path, relative_label: Path) -> Path | None:
    """Tìm ảnh tương ứng với một label YOLO theo stem và extension hỗ trợ."""
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
    """Trả về label text đã remap và ghi nhận lỗi với class id không hợp lệ."""

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


def stratified_split(
    rows: Sequence[dict[str, str]],
    label_key: str = "label",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, str]]]:
    """Tạo stratified split ổn định và giữ thứ tự input trong từng label."""

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and test > 0")
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[label_key], []).append(row)

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        total = len(label_rows)
        train_end = max(1, int(total * train_ratio)) if total > 1 else total
        val_end = train_end + int(total * val_ratio)
        splits["train"].extend(label_rows[:train_end])
        splits["val"].extend(label_rows[train_end:val_end])
        splits["test"].extend(label_rows[val_end:])
    return splits


def group_aware_split(
    rows: Sequence[dict[str, str]],
    group_key: str = "group",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, str]]]:
    """Tạo split ổn định, đảm bảo mỗi group chỉ nằm trong một split."""

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    ordered_groups = sorted(groups)
    total = len(ordered_groups)
    train_end = max(1, int(total * train_ratio)) if total > 1 else total
    val_end = train_end + int(total * val_ratio)
    split_groups = {
        "train": set(ordered_groups[:train_end]),
        "val": set(ordered_groups[train_end:val_end]),
        "test": set(ordered_groups[val_end:]),
    }
    return {
        split: [row for row in rows if row[group_key] in group_names]
        for split, group_names in split_groups.items()
    }


def build_hmdb_manifest(
    videos_root: Path,
    class_names: Sequence[str],
    split_root: Path | None = None,
    split_index: int = 1,
) -> list[dict[str, str]]:
    """Tạo manifest HMDB-style từ thư mục video phân cấp theo class."""

    allowed_classes = set(class_names)
    split_by_file = (
        read_hmdb_split_assignments(split_root, split_index)
        if split_root is not None and split_root.exists()
        else {}
    )
    rows: list[dict[str, str]] = []
    for video_path in sorted(videos_root.rglob("*")):
        if (
            not video_path.is_file()
            or video_path.suffix.lower() not in VIDEO_EXTENSIONS
        ):
            continue
        label = video_path.parent.name
        if label not in allowed_classes:
            continue
        rows.append(
            {
                "path": str(video_path),
                "relative_path": video_path.relative_to(videos_root).as_posix(),
                "label": label,
                "group": video_path.stem,
                "split": split_by_file.get(video_path.name, "unassigned"),
            }
        )
    return rows


def read_hmdb_split_assignments(
    split_root: Path,
    split_index: int = 1,
) -> dict[str, str]:
    """Đọc các file HMDB `*_test_splitN.txt` thành mapping `file -> split`."""

    assignments: dict[str, str] = {}
    split_labels = {"0": "unused", "1": "train", "2": "test"}
    for split_path in sorted(split_root.glob(f"*_test_split{split_index}.txt")):
        for line in split_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            assignments[parts[0]] = split_labels.get(parts[1], "unknown")
    return assignments


def sample_frame_indices(
    frame_count: int,
    sequence_length: int = 16,
    frame_stride: int = 2,
) -> tuple[int, ...]:
    """Lấy chuỗi frame ở giữa video theo `sequence_length` và `frame_stride`."""

    if frame_count <= 0 or sequence_length <= 0 or frame_stride <= 0:
        raise ValueError(
            "frame_count, sequence_length, and frame_stride must be positive"
        )
    required_span = (sequence_length - 1) * frame_stride + 1
    if frame_count < required_span:
        return ()
    start = (frame_count - required_span) // 2
    return tuple(start + index * frame_stride for index in range(sequence_length))


def add_frame_sequence_columns(
    rows: Sequence[dict[str, str]],
    frame_counts: dict[str, int],
    sequence_length: int = 16,
    frame_stride: int = 2,
) -> list[dict[str, str]]:
    """Bổ sung cột frame sequence cho các video có đủ số frame."""

    sequence_rows: list[dict[str, str]] = []
    for row in rows:
        path = row["path"]
        frame_count = frame_counts.get(path)
        if frame_count is None:
            continue
        indices = sample_frame_indices(frame_count, sequence_length, frame_stride)
        if not indices:
            continue
        sequence_rows.append(
            {
                **row,
                "frame_count": str(frame_count),
                "frame_indices": " ".join(str(index) for index in indices),
                "sequence_length": str(sequence_length),
                "frame_stride": str(frame_stride),
            }
        )
    return sequence_rows


def validate_video_manifest(
    rows: Sequence[dict[str, str]],
    class_names: Sequence[str],
) -> ValidationReport:
    """Validate manifest video cơ bản cho action pipeline."""

    allowed_classes = set(class_names)
    issues: list[ValidationIssue] = []
    group_to_splits: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=1):
        path = Path(row.get("path", ""))
        label = row.get("label", "")
        split = row.get("split", "")
        group = row.get("group", path.stem)
        row_path = str(path) if str(path) else f"row:{index}"

        if not path.exists():
            issues.append(ValidationIssue("error", row_path, "missing video file"))
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            issues.append(
                ValidationIssue("error", row_path, "unsupported video extension")
            )
        if label not in allowed_classes:
            issues.append(ValidationIssue("error", row_path, f"unknown class: {label}"))
        if split and split not in {"train", "val", "test", "unused", "unassigned"}:
            issues.append(ValidationIssue("error", row_path, f"unknown split: {split}"))
        if split not in {"", "unused", "unassigned"}:
            group_to_splits.setdefault(group, set()).add(split)

    for group, splits in group_to_splits.items():
        if len(splits) > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    group,
                    f"group appears in multiple splits: {sorted(splits)}",
                )
            )
    return ValidationReport(len(rows), tuple(issues))


def write_manifest_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    """Ghi manifest dạng CSV với fieldnames ổn định từ các row."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_hmdb_classes(root: Path) -> list[str]:
    """Trả về tên class HMDB từ các thư mục class."""

    return sorted(path.name for path in root.iterdir() if path.is_dir())


def parse_places_categories(path: Path) -> dict[str, int]:
    """Parse file categories Places365 thành mapping `category -> index`."""

    categories: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        category = parts[0].lstrip("/")
        index = (
            int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else len(categories)
        )
        categories[category] = index
    return categories


def parse_msr_vtt_captions(path: Path) -> dict[str, list[str]]:
    """Normalize annotation kiểu MSR-VTT thành mapping `video_id -> captions`."""

    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", data if isinstance(data, list) else [])
    if not isinstance(annotations, list):
        raise ValueError("Expected MSR-VTT annotations list")
    captions: dict[str, list[str]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        video_id = annotation.get("video_id", annotation.get("image_id"))
        caption = annotation.get("caption", annotation.get("sentence"))
        if video_id is None or caption is None:
            continue
        captions.setdefault(str(video_id), []).append(str(caption))
    return captions


def parse_icdar_gt(path: Path) -> list[IcdarTextBox]:
    """Parse ground truth localization/transcription của ICDAR 2015."""

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


def validate_wav(
    path: Path,
    expected_sample_rate_hz: int | None = None,
) -> ValidationReport:
    """Kiểm tra file WAV được synthesize từ Piper."""

    issues: list[ValidationIssue] = []
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
    except wave.Error as exc:
        return ValidationReport(1, (ValidationIssue("error", str(path), str(exc)),))

    if frames <= 0:
        issues.append(ValidationIssue("error", str(path), "duration is zero"))
    if channels != 1:
        issues.append(ValidationIssue("error", str(path), "expected mono audio"))
    if sample_width not in {2, 4}:
        issues.append(ValidationIssue("error", str(path), "expected PCM sample width"))
    if expected_sample_rate_hz is not None and sample_rate != expected_sample_rate_hz:
        issues.append(
            ValidationIssue(
                "error",
                str(path),
                f"sample rate {sample_rate} != {expected_sample_rate_hz}",
            )
        )
    return ValidationReport(1, tuple(issues))


def tts_voice_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Trả về danh sách voice Piper mặc định từ TTS config."""

    voices = config.get("voices", {})
    if not isinstance(voices, dict):
        raise ValueError("TTS config missing voices")
    defaults = voices.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("TTS config missing voices.defaults")
    return [entry for entry in defaults.values() if isinstance(entry, dict)]


def validate_tts_config(
    config_path: Path,
    execute: bool = False,
    strict_files: bool = True,
) -> ValidationReport:
    """Kiểm tra config voice Piper và tùy chọn synthesize WAV sample."""

    config = load_yaml(config_path)
    issues: list[ValidationIssue] = []
    checked_files = 0
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("TTS config missing paths")
    output_root = Path(str(paths.get("validation_output", "data/tts_validation")))

    for voice in tts_voice_entries(config):
        model_path = Path(str(voice["model_path"]))
        voice_config_path = Path(str(voice["config_path"]))
        checked_files += 2
        if not model_path.exists() and strict_files:
            issues.append(
                ValidationIssue("error", str(model_path), "missing model file")
            )
        if not voice_config_path.exists():
            if strict_files:
                issues.append(
                    ValidationIssue(
                        "error",
                        str(voice_config_path),
                        "missing config file",
                    )
                )
            continue
        try:
            json.loads(voice_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue("error", str(voice_config_path), f"invalid json: {exc}")
            )
        if execute and model_path.exists() and voice_config_path.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            output_wav = output_root / f"{voice['id']}.wav"
            synthesize_with_piper(
                model_path=model_path,
                output_wav=output_wav,
                text=str(voice["prompt"]),
            )
            checked_files += 1
            issues.extend(
                validate_wav(
                    output_wav,
                    expected_sample_rate_hz=int(voice["expected_sample_rate_hz"]),
                ).issues
            )
    return ValidationReport(checked_files, tuple(issues))


def synthesize_with_piper(model_path: Path, output_wav: Path, text: str) -> None:
    """Chạy Piper CLI cho một prompt."""

    command = [
        "piper",
        "--model",
        str(model_path),
        "--output_file",
        str(output_wav),
    ]
    subprocess.run(
        command,
        input=text,
        text=True,
        check=True,
        capture_output=True,
    )


def augmentation_preset(task: str, size: int = 640) -> AugmentationPreset:
    """Trả về metadata preset augmentation cho image task."""

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
    """Tạo hoặc preview sample augmentation cho image dataset."""

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


def read_classes(path: Path) -> list[str]:
    """Đọc danh sách tên class từ file text."""

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_split_csvs(
    splits: dict[str, list[dict[str, str]]],
    output_dir: Path,
    fieldnames: Sequence[str],
) -> None:
    """Ghi từng split ra file CSV riêng trong thư mục output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (output_dir / f"{split_name}.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo parser dòng lệnh cho các tiện ích data pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-plan")
    download.add_argument("--config-dir", type=Path, default=DEFAULT_DATASET_CONFIG_DIR)
    download.add_argument("--execute", action="store_true")

    split = subparsers.add_parser("split")
    split.add_argument("--input-csv", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--label-key", default="label")
    split.add_argument("--group-key")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--classes", type=Path, required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--coco-json", type=Path, required=True)
    convert.add_argument("--image-root", type=Path, required=True)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--classes", type=Path, required=True)
    convert.add_argument("--split", default="train")
    convert.add_argument("--execute", action="store_true")

    merge = subparsers.add_parser("merge-yolo")
    merge.add_argument("--sources", type=Path, nargs="+", required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--classes", type=Path, required=True)
    merge.add_argument("--dry-run", action="store_true")

    augment = subparsers.add_parser("augment-preview")
    augment.add_argument("--input-dir", type=Path, required=True)
    augment.add_argument("--output-dir", type=Path, required=True)
    augment.add_argument(
        "--task",
        choices=["detection", "scene", "ocr"],
        default="detection",
    )
    augment.add_argument("--size", type=int, default=640)
    augment.add_argument("--limit", type=int, default=8)
    augment.add_argument("--execute", action="store_true")

    action_manifest = subparsers.add_parser("build-action-manifest")
    action_manifest.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/action_accessibility.yaml"),
    )
    action_manifest.add_argument("--videos-root", type=Path)
    action_manifest.add_argument("--split-root", type=Path)
    action_manifest.add_argument("--output-csv", type=Path)
    action_manifest.add_argument("--split-index", type=int, default=1)
    action_manifest.add_argument("--execute", action="store_true")

    validate_tts_parser = subparsers.add_parser("validate-tts")
    validate_tts_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/tts_piper_accessibility.yaml"),
    )
    validate_tts_parser.add_argument("--execute", action="store_true")
    validate_tts_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI data pipeline và trả về mã thoát."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "download-plan":
        if args.execute:
            print("Execute mode is reserved for explicit dataset download wrappers.")
        print("\n".join(download_plan_lines(dataset_config_paths(args.config_dir))))
        return 0

    if args.command == "split":
        with args.input_csv.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"No rows found in {args.input_csv}")
        splits = (
            group_aware_split(rows, group_key=args.group_key)
            if args.group_key
            else stratified_split(rows, label_key=args.label_key)
        )
        write_split_csvs(splits, args.output_dir, fieldnames=list(rows[0].keys()))
        print({name: len(rows_for_split) for name, rows_for_split in splits.items()})
        return 0

    if args.command == "validate":
        validation_report = validate_yolo_dataset(
            args.dataset_root,
            read_classes(args.classes),
        )
        print_report(validation_report)
        return 0 if validation_report.valid else 1

    if args.command == "convert":
        count = write_yolo_from_coco(
            annotation_path=args.coco_json,
            image_root=args.image_root,
            output_root=args.output_root,
            class_names=read_classes(args.classes),
            split=args.split,
            dry_run=not args.execute,
        )
        print(f"selected_images={count} dry_run={not args.execute}")
        return 0

    if args.command == "merge-yolo":
        merge_report = merge_yolo_datasets(
            sources=args.sources,
            output_root=args.output_root,
            canonical_names=read_classes(args.classes),
            dry_run=args.dry_run,
        )
        print_merge_report(merge_report)
        return 0 if merge_report.valid else 1

    if args.command == "augment-preview":
        count = augment_preview(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            dry_run=not args.execute,
            task=args.task,
            size=args.size,
        )
        print(
            " ".join(
                [
                    f"augmented_or_previewed={count}",
                    f"task={args.task}",
                    f"size={args.size}",
                    f"dry_run={not args.execute}",
                ]
            )
        )
        return 0

    if args.command == "build-action-manifest":
        config = load_yaml(args.config)
        base_dataset = config.get("base_dataset", {})
        classes = config.get("classes", {})
        output = config.get("output", {})
        if not isinstance(base_dataset, dict) or not isinstance(classes, dict):
            raise ValueError("Action config missing base_dataset/classes")
        if not isinstance(output, dict):
            raise ValueError("Action config missing output")
        public_classes = classes.get("public_bootstrap", [])
        if not isinstance(public_classes, list):
            raise ValueError("Action config classes.public_bootstrap must be a list")
        videos_root = args.videos_root or Path(str(base_dataset["videos_root"]))
        split_root = args.split_root or Path(str(base_dataset["split_root"]))
        output_csv = args.output_csv or Path(str(output["manifest"]))
        rows = build_hmdb_manifest(
            videos_root=videos_root,
            class_names=[str(name) for name in public_classes],
            split_root=split_root,
            split_index=args.split_index,
        )
        report = validate_video_manifest(rows, [str(name) for name in public_classes])
        print_report(report)
        if args.execute and report.valid:
            write_manifest_csv(rows, output_csv)
            print(f"wrote_manifest={output_csv} rows={len(rows)}")
        else:
            print(f"planned_manifest={output_csv} rows={len(rows)} dry_run=True")
        return 0

    if args.command == "validate-tts":
        tts_report = validate_tts_config(
            args.config,
            execute=args.execute,
            strict_files=not args.dry_run,
        )
        print_report(tts_report)
        return 0 if tts_report.valid else 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


def print_report(report: ValidationReport) -> None:
    """In validation report cho CLI."""

    print(f"checked_files={report.checked_files} valid={report.valid}")
    for issue in report.issues:
        print(f"{issue.level}: {issue.path}: {issue.message}")


def print_merge_report(report: MergeReport) -> None:
    """In merge report cho CLI."""

    print(
        " ".join(
            [
                f"dry_run={report.dry_run}",
                f"scanned_labels={report.scanned_labels}",
                f"copied_files={report.copied_files}",
                f"valid={report.valid}",
            ]
        )
    )
    print(json.dumps(report.class_remap, indent=2, sort_keys=True))
    for issue in report.issues:
        print(f"{issue.level}: {issue.path}: {issue.message}")


if __name__ == "__main__":
    sys.exit(main())
