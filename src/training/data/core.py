# -*- coding: utf-8 -*-
"""Kiểu dữ liệu và tiện ích dùng chung cho data pipeline."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
DEFAULT_DATASET_CONFIG_DIR = Path("configs/datasets")
DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ValidationIssue:
    """Một lỗi hoặc cảnh báo trong quá trình validate dataset."""

    level: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Báo cáo tổng hợp từ các hàm validation."""

    checked_files: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """Cho biết report không có issue cấp `error`."""
        return not any(issue.level == "error" for issue in self.issues)


@dataclass(frozen=True)
class MergeReport:
    """Báo cáo preview hoặc execute khi merge dataset YOLO."""

    dry_run: bool
    scanned_labels: int
    copied_files: int
    class_remap: dict[str, dict[int, int]]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """Cho biết merge report không có issue cấp `error`."""
        return not any(issue.level == "error" for issue in self.issues)


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML và trả về mapping."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def dataset_config_paths(config_dir: Path = DEFAULT_DATASET_CONFIG_DIR) -> list[Path]:
    """Liệt kê dataset config YAML theo thứ tự ổn định."""

    return sorted(config_dir.glob("*.yaml"))


def write_manifest_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    """Ghi manifest CSV và tự suy ra fieldnames từ rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_classes(path: Path) -> list[str]:
    """Đọc danh sách canonical class names từ file text."""

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
    """Ghi từng split subset ra file CSV trong output directory."""
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
