"""Tiện ích chia dataset ổn định."""

from __future__ import annotations

from collections.abc import Sequence


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
    """Tạo group-aware split sao cho mỗi group chỉ thuộc một split."""

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
