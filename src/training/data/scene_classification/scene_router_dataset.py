"""Dataset builders for scene router labels."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from src.training.data.core import IMAGE_EXTENSIONS
from src.training.data.ocr import parse_textocr_annotations

SCENE_ROUTER_LABELS: tuple[str, ...] = (
    "static",
    "motion",
    "text_present",
    "face_detected",
    "scene_change",
)


def build_scene_router_manifest(
    static_rows: Sequence[dict[str, str]],
    motion_rows: Sequence[dict[str, str]],
    text_rows: Sequence[dict[str, str]],
    face_rows: Sequence[dict[str, str]],
    scene_change_rows: Sequence[dict[str, str]],
    *,
    min_samples_per_class: int = 500,
    max_samples_per_class: int = 1000,
    seed: int = 501,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[dict[str, str]]:
    """Merge class-specific rows into a balanced `path,label,source,split` manifest."""

    if min_samples_per_class <= 0:
        raise ValueError("min_samples_per_class must be positive")
    if max_samples_per_class < min_samples_per_class:
        raise ValueError("max_samples_per_class must be >= min_samples_per_class")
    grouped = {
        "static": list(static_rows),
        "motion": list(motion_rows),
        "text_present": list(text_rows),
        "face_detected": list(face_rows),
        "scene_change": list(scene_change_rows),
    }
    rng = random.Random(seed)
    output_rows: list[dict[str, str]] = []
    for label in SCENE_ROUTER_LABELS:
        rows = grouped[label]
        if len(rows) < min_samples_per_class:
            raise ValueError(
                f"class {label} has only {len(rows)} rows, expected at least "
                f"{min_samples_per_class}"
            )
        selected = _sample_rows(rows, max_samples_per_class, rng)
        splits = split_names(len(selected), train_ratio=train_ratio, val_ratio=val_ratio)
        for row, split in zip(selected, splits, strict=True):
            output_rows.append(
                {
                    "path": row["path"],
                    "label": label,
                    "source": row["source"],
                    "split": split,
                }
            )
    return output_rows


def collect_static_rows(
    scene_manifest_csv: Path,
    max_samples: int,
) -> list[dict[str, str]]:
    """Collect `static` candidates from Places-derived scene manifest."""

    rows = []
    for row in read_csv_rows(scene_manifest_csv):
        image_path = Path(row.get("path", ""))
        if image_path.exists() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": str(image_path), "source": "places365"})
            if len(rows) >= max_samples:
                break
    return rows


def collect_motion_rows(
    frame_manifest_csv: Path,
    max_samples: int,
) -> list[dict[str, str]]:
    """Collect `motion` candidates from UCF frame directories."""

    rows = []
    for row in read_csv_rows(frame_manifest_csv):
        frame_dir = Path(row.get("frame_dir", ""))
        if not frame_dir.exists():
            continue
        frame = _pick_frame_from_dir(frame_dir, index=1)
        if frame is None:
            continue
        rows.append({"path": str(frame), "source": "ucf101"})
        if len(rows) >= max_samples:
            break
    return rows


def collect_scene_change_rows(
    frame_manifest_csv: Path,
    max_samples: int,
    diff_threshold: float = 20.0,
) -> list[dict[str, str]]:
    """Collect scene-change candidates from UCF frame pairs with strong difference."""

    rows = []
    for row in read_csv_rows(frame_manifest_csv):
        frame_dir = Path(row.get("frame_dir", ""))
        if not frame_dir.exists():
            continue
        frames = sorted(
            path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(frames) < 2:
            continue
        first, last = frames[0], frames[-1]
        if frame_difference_score(first, last) < diff_threshold:
            continue
        rows.append({"path": str(last), "source": "ucf101_scene_change"})
        if len(rows) >= max_samples:
            break
    return rows


def collect_text_present_rows(
    annotations: Sequence[Path],
    image_roots: Sequence[Path],
    max_samples: int,
) -> list[dict[str, str]]:
    """Collect positive text images from TextOCR-style annotations."""

    rows = []
    seen: set[Path] = set()
    for annotation in annotations:
        if not annotation.exists():
            continue
        for box in parse_textocr_annotations(annotation):
            if box.ignored or not box.image_file:
                continue
            image_path = _resolve_image_path(box.image_file, image_roots)
            if image_path is None or image_path in seen:
                continue
            seen.add(image_path)
            rows.append({"path": str(image_path), "source": "textocr"})
            if len(rows) >= max_samples:
                return rows
    return rows


def collect_face_detected_rows(
    face_roots: Sequence[Path],
    max_samples: int,
) -> list[dict[str, str]]:
    """Collect `face_detected` candidates from face datasets folders."""

    rows = []
    for root in face_roots:
        if not root.exists():
            continue
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append({"path": str(image_path), "source": root.name})
            if len(rows) >= max_samples:
                return rows
    return rows


def augment_balance_plan(
    rows: Sequence[dict[str, str]],
    target_per_class: int,
) -> dict[str, int]:
    """Return required augmentation counts per label to reach target count."""

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1
    return {
        label: max(0, target_per_class - counts.get(label, 0))
        for label in SCENE_ROUTER_LABELS
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read rows from a CSV file."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_scene_router_manifest(rows: Sequence[dict[str, str]], output_csv: Path) -> None:
    """Write the unified scene router manifest."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)


def split_names(total: int, train_ratio: float, val_ratio: float) -> list[str]:
    """Return deterministic split names for a class bucket."""

    if total <= 0:
        return []
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("invalid split ratio configuration")
    test_ratio = 1.0 - train_ratio - val_ratio
    targets = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [math.floor(target) for target in targets]
    remainder = total - sum(counts)
    order = sorted(
        range(len(targets)),
        key=lambda index: (targets[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    if total >= 3:
        for index in range(3):
            if counts[index] == 0 and targets[index] > 0:
                donor = max(range(3), key=lambda donor_index: counts[donor_index])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    train_count, val_count, test_count = counts
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def frame_difference_score(path_a: Path, path_b: Path) -> float:
    """Estimate frame difference with mean absolute RGB delta."""

    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        rgb_a = image_a.convert("RGB")
        rgb_b = image_b.convert("RGB").resize(rgb_a.size)
        delta = ImageChops.difference(rgb_a, rgb_b)
        stat = ImageStat.Stat(delta)
    return float(sum(stat.mean) / len(stat.mean))


def _sample_rows(
    rows: Sequence[dict[str, str]],
    max_samples: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if len(rows) <= max_samples:
        return list(rows)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[index] for index in indices[:max_samples]]


def _pick_frame_from_dir(frame_dir: Path, index: int) -> Path | None:
    frames = sorted(
        path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        return None
    return frames[min(index, len(frames) - 1)]


def _resolve_image_path(relative_path: str, roots: Iterable[Path]) -> Path | None:
    relative = Path(relative_path.lstrip("/"))
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
        fallback = root / relative.name
        if fallback.exists():
            return fallback
    return None
