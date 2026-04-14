# -*- coding: utf-8 -*-
"""Shared types and helpers for the data pipeline."""

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


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc file YAML từ đĩa và trả về mapping."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def dataset_config_paths(config_dir: Path = DEFAULT_DATASET_CONFIG_DIR) -> list[Path]:
    """Trả về danh sách file config dataset theo thứ tự ổn định."""

    return sorted(config_dir.glob("*.yaml"))


def write_manifest_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    """Ghi manifest dạng CSV với fieldnames ổn định từ các row."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
