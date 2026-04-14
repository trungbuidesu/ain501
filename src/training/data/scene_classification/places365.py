"""Tiện ích scene classification cho Places365.

Module tạo balanced manifest từ Places-style folders hoặc filelists. Việc copy
ảnh sang subset root tách riêng khỏi bước tạo manifest.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from src.training.data.core import IMAGE_EXTENSIONS


def build_scene_subset_manifest(
    images_root: Path,
    class_names: Sequence[str],
    max_images_per_class: int = 1000,
) -> list[dict[str, str]]:
    """Tạo balanced scene manifest từ local category folders.

    Mỗi row chứa source path, relative path, label và source category. Số ảnh
    mỗi class không vượt `max_images_per_class`.
    """

    if max_images_per_class <= 0:
        raise ValueError("max_images_per_class must be positive")
    rows: list[dict[str, str]] = []
    for class_name in class_names:
        matched = 0
        for image_path in sorted(images_root.rglob("*")):
            if matched >= max_images_per_class:
                break
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue
            relative_parent = image_path.parent.relative_to(images_root).as_posix()
            if not _scene_path_matches_class(relative_parent, class_name):
                continue
            rows.append(
                {
                    "path": str(image_path),
                    "relative_path": image_path.relative_to(images_root).as_posix(),
                    "label": class_name,
                    "source_category": relative_parent,
                }
            )
            matched += 1
    return rows


def build_scene_subset_manifest_from_places_filelist(
    images_root: Path,
    filelist_path: Path,
    categories_file: Path,
    class_names: Sequence[str],
    max_images_per_class: int = 1000,
) -> list[dict[str, str]]:
    """Tạo balanced manifest từ Places365 filelist metadata.

    Dùng `categories_places365.txt` để map class index sang category path, sau
    đó lọc theo configured scene classes.
    """

    if max_images_per_class <= 0:
        raise ValueError("max_images_per_class must be positive")
    categories = read_places_categories(categories_file)
    counts = dict.fromkeys(class_names, 0)
    rows: list[dict[str, str]] = []
    with filelist_path.open(encoding="utf-8") as file:
        for line in file:
            parsed = _parse_places_filelist_line(line)
            if parsed is None:
                continue
            relative_path, category_index = parsed
            source_category = categories.get(category_index)
            if source_category is None:
                continue
            for class_name in class_names:
                if counts[class_name] >= max_images_per_class:
                    continue
                if not _scene_path_matches_class(source_category, class_name):
                    continue
                image_path = _resolve_places_image_path(images_root, relative_path)
                rows.append(
                    {
                        "path": str(image_path),
                        "relative_path": (
                            image_path.relative_to(images_root).as_posix()
                            if image_path.is_relative_to(images_root)
                            else relative_path
                        ),
                        "label": class_name,
                        "source_category": source_category,
                    }
                )
                counts[class_name] += 1
                break
            if all(count >= max_images_per_class for count in counts.values()):
                break
    return rows


def read_places_categories(categories_file: Path) -> dict[int, str]:
    """Đọc `categories_places365.txt` thành mapping index -> category."""

    categories: dict[int, str] = {}
    with categories_file.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                index = int(parts[-1])
            except ValueError:
                continue
            categories[index] = parts[0].strip("/")
    return categories


def _parse_places_filelist_line(line: str) -> tuple[str, int] | None:
    """Parse một dòng Places365 filelist."""

    parts = line.strip().split()
    if len(parts) < 2:
        return None
    try:
        category_index = int(parts[-1])
    except ValueError:
        return None
    return parts[0].lstrip("/"), category_index


def _resolve_places_image_path(images_root: Path, relative_path: str) -> Path:
    """Resolve path cho layout train folder hoặc flat val archive."""

    relative = Path(relative_path)
    candidates = [
        images_root / relative,
        images_root / relative.name,
        images_root / "val_256" / relative.name,
        images_root / "data_256_standard" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _find_default_places_filelist(root: Path) -> Path | None:
    """Tìm default Places365 filelist khi caller không truyền path."""

    candidates = [
        root / "places365_train_standard.txt",
        root / "places365_val.txt",
        root / "filelist_places365-standard" / "places365_train_standard.txt",
        root / "filelist_places365-standard" / "places365_val.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _scene_path_matches_class(relative_parent: str, class_name: str) -> bool:
    """Khớp class name với category path kiểu Places."""

    normalized_parent = relative_parent.replace("\\", "/").strip("/")
    normalized_class = class_name.strip("/")
    if "/" in normalized_class:
        return normalized_parent == normalized_class or normalized_parent.endswith(
            f"/{normalized_class}"
        )
    return (
        normalized_parent == normalized_class
        or normalized_parent.endswith(f"/{normalized_class}")
        or normalized_parent.split("/")[-1] == normalized_class.split("/")[-1]
    )


def copy_scene_subset(
    rows: Sequence[dict[str, str]],
    subset_root: Path,
) -> int:
    """Copy ảnh scene subset theo label vào `subset_root`."""

    copied = 0
    for row in rows:
        source = Path(row["path"])
        label = row["label"].replace("/", "_")
        destination = subset_root / "images" / label / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def parse_places_categories(path: Path) -> dict[str, int]:
    """Parse category file thành mapping category -> index."""

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
