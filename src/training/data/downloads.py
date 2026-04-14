# -*- coding: utf-8 -*-
"""Tiện ích tạo download plan và verify path dataset local."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.training.data.core import ValidationIssue, ValidationReport, load_yaml


def download_plan_lines(config_paths: Sequence[Path]) -> list[str]:
    """Tạo các dòng download plan từ dataset configs."""

    lines: list[str] = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        lines.append(f"# {config.get('name', config_path.stem)}")
        _append_download_details(config, lines)
        lines.append("")
    return lines


def _append_download_details(node: Any, lines: list[str]) -> None:
    """Duyệt config node và append thông tin download vào output lines."""
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


def verify_dataset_paths(
    config_paths: Sequence[Path],
    tasks: Sequence[str] | None = None,
    include_outputs: bool = False,
    include_downloads: bool = True,
) -> ValidationReport:
    """Verify các local paths quan trọng được khai báo trong dataset configs."""

    task_filter = set(tasks or [])
    issues: list[ValidationIssue] = []
    checked_files = 0
    for config_path in config_paths:
        config = load_yaml(config_path)
        task = str(config.get("task", ""))
        if task_filter and task not in task_filter:
            continue
        nodes: list[tuple[str, Any]] = []
        if "base_dataset" in config:
            nodes.append(("base_dataset", config["base_dataset"]))
        if include_outputs and "output" in config:
            nodes.append(("output", config["output"]))
        for node_name, node in nodes:
            for key_path, path in _declared_local_paths(
                node,
                prefix=node_name,
                include_downloads=include_downloads,
            ):
                checked_files += 1
                if path.exists():
                    continue
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{config_path}:{key_path}",
                        f"missing local path: {path}",
                    )
                )
    return ValidationReport(checked_files, tuple(issues))


def _declared_local_paths(
    node: Any,
    prefix: str = "",
    include_downloads: bool = True,
) -> list[tuple[str, Path]]:
    """Trích xuất local paths từ config node và bỏ qua URL paths."""

    paths: list[tuple[str, Path]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if not include_downloads and ".download." in f".{key_path}.":
                continue
            if key in {"homepage", "url", "manual_note", "variant", "name", "task"}:
                continue
            if isinstance(value, str) and _looks_like_local_path(value):
                paths.append((key_path, Path(value)))
            elif isinstance(value, (dict, list)):
                paths.extend(
                    _declared_local_paths(
                        value,
                        key_path,
                        include_downloads=include_downloads,
                    )
                )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            paths.extend(
                _declared_local_paths(
                    value,
                    f"{prefix}[{index}]",
                    include_downloads=include_downloads,
                )
            )
    return paths


def _looks_like_local_path(value: str) -> bool:
    """Heuristic nhận diện chuỗi có vẻ là local path."""

    if "://" in value or value.startswith("Phase "):
        return False
    path_markers = {"/", "\\"}
    suffixes = {".json", ".txt", ".csv", ".yaml", ".yml", ".zip", ".jsonl"}
    return any(marker in value for marker in path_markers) or any(
        value.endswith(suffix) for suffix in suffixes
    )
