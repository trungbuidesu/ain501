# -*- coding: utf-8 -*-
"""Tiện ích nhận diện hành động cho UCF-101.

Module dựng manifest từ thư mục video theo class, đọc split chính thức
UCF-101, lấy mẫu frame sequence tiền định và kiểm tra bất biến group/split cho
pipeline Phase 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.training.data.core import (
    VIDEO_EXTENSIONS,
    ValidationIssue,
    ValidationReport,
    load_yaml,
)


def build_action_manifest(
    videos_root: Path,
    class_names: Sequence[str],
    dataset_name: str = "class_folders",
    split_root: Path | None = None,
    split_index: int = 1,
) -> list[dict[str, str]]:
    """Tạo action manifest từ thư mục video theo class.

    Mỗi row lưu path tuyệt đối, relative path, label, group và split. Chỉ các
    class nằm trong `class_names` được đưa vào manifest.
    """

    allowed_classes = set(class_names)
    split_by_key = read_action_split_assignments(
        dataset_name,
        split_root,
        split_index=split_index,
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
        relative_path = video_path.relative_to(videos_root).as_posix()
        rows.append(
            {
                "path": str(video_path),
                "relative_path": relative_path,
                "label": label,
                "group": video_path.stem,
                "split": split_by_key.get(
                    relative_path,
                    split_by_key.get(video_path.name, "unassigned"),
                ),
            }
        )
    return rows


def read_action_split_assignments(
    dataset_name: str,
    split_root: Path | None,
    split_index: int = 1,
) -> dict[str, str]:
    """Đọc split assignment cho action dataset theo cấu hình."""

    if split_root is None or not split_root.exists():
        return {}
    normalized_name = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized_name == "ucf101":
        return read_ucf101_split_assignments(split_root, split_index)
    return {}


def read_ucf101_split_assignments(
    split_root: Path,
    split_index: int = 1,
) -> dict[str, str]:
    """Đọc train/test list chính thức của UCF-101.

    Trả về mapping từ `ClassName/video.avi` sang `train` hoặc `test`. Thiếu
    file split được bỏ qua để dry-run vẫn chạy được trước khi có data local.
    """

    assignments: dict[str, str] = {}
    train_path = split_root / f"trainlist{split_index:02d}.txt"
    test_path = split_root / f"testlist{split_index:02d}.txt"
    if train_path.exists():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                assignments[parts[0].replace("\\", "/")] = "train"
    if test_path.exists():
        for line in test_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                assignments[parts[0].replace("\\", "/")] = "test"
    return assignments


def sample_frame_indices(
    frame_count: int,
    sequence_length: int = 16,
    frame_stride: int = 2,
) -> tuple[int, ...]:
    """Lấy mẫu frame sequence centered và tiền định.

    Trả về tuple rỗng khi video quá ngắn so với span yêu cầu để caller có thể
    bỏ qua clip ngắn mà không raise.
    """

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
    """Bổ sung cột frame sequence cho video có đủ frame."""

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
    check_video_open: bool = False,
    min_frames: int = 1,
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

        path_exists = path.exists()
        supported_extension = path.suffix.lower() in VIDEO_EXTENSIONS
        if not path_exists:
            issues.append(ValidationIssue("error", row_path, "missing video file"))
        if not supported_extension:
            issues.append(
                ValidationIssue("error", row_path, "unsupported video extension")
            )
        if check_video_open and path_exists and supported_extension:
            open_issue = validate_video_open(path, min_frames=min_frames)
            if open_issue is not None:
                issues.append(open_issue)
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


def validate_video_open(path: Path, min_frames: int = 1) -> ValidationIssue | None:
    """Mở video bằng OpenCV và kiểm tra số frame tối thiểu."""

    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return ValidationIssue(
            "error",
            str(path),
            "opencv-python is required for video open validation",
        )

    capture = cv2.VideoCapture(str(path))
    try:
        if not bool(capture.isOpened()):
            return ValidationIssue("error", str(path), "cannot open video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count < min_frames:
            return ValidationIssue(
                "error",
                str(path),
                f"video has {frame_count} frames, expected at least {min_frames}",
            )
    finally:
        capture.release()
    return None


def read_action_classes(
    config_path: Path,
    class_key: str = "public_bootstrap",
) -> list[str]:
    """Đọc whitelist class nhận diện hành động từ config."""

    config = load_yaml(config_path)
    classes = config.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("Action config missing classes mapping")
    names = classes.get(class_key, [])
    if not isinstance(names, list):
        raise ValueError(f"Action config classes.{class_key} must be a list")
    return [str(name) for name in names]
