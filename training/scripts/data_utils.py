"""Dataset preparation utilities for Phase 0 accessibility data work."""

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
DEFAULT_DATASET_CONFIG_DIR = Path("configs/datasets")


@dataclass(frozen=True)
class ValidationIssue:
    """A single dataset validation issue."""

    level: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Summary returned by validators."""

    checked_files: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


@dataclass(frozen=True)
class MergeReport:
    """Summary returned by YOLO merge preview/execution."""

    dry_run: bool
    scanned_labels: int
    copied_files: int
    class_remap: dict[str, dict[int, int]]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


@dataclass(frozen=True)
class IcdarTextBox:
    """ICDAR text localization entry."""

    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str
    ignored: bool


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def dataset_config_paths(config_dir: Path = DEFAULT_DATASET_CONFIG_DIR) -> list[Path]:
    """Return dataset config files in deterministic order."""

    return sorted(config_dir.glob("*.yaml"))


def download_plan_lines(config_paths: Sequence[Path]) -> list[str]:
    """Build a human-readable download plan from dataset configs."""

    lines: list[str] = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        lines.append(f"# {config.get('name', config_path.stem)}")
        _append_download_details(config, lines)
        lines.append("")
    return lines


def _append_download_details(node: Any, lines: list[str]) -> None:
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
    """Convert COCO ``x, y, width, height`` pixels to normalized YOLO bbox."""

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
    """Convert selected COCO annotations to YOLO label lines keyed by image file."""

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
    """Convert a COCO annotation JSON into a YOLO directory for one split."""

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
    """Validate a YOLO image/label dataset rooted at ``root``."""

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
    """Validate a single normalized YOLO label line."""

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
    """Merge YOLO datasets while remapping each source's class ids."""

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
    """Read YOLO class names from classes.txt or Ultralytics data.yaml."""

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
    """Return remapped label text, recording issues for unknown ids."""

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
    """Deterministic stratified split that preserves input order within labels."""

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
    """Deterministic split that keeps each group in only one split."""

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


def parse_hmdb_classes(root: Path) -> list[str]:
    """Return HMDB class names from class directories."""

    return sorted(path.name for path in root.iterdir() if path.is_dir())


def parse_places_categories(path: Path) -> dict[str, int]:
    """Parse a Places365 categories file into ``category -> index``."""

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
    """Normalize MSR-VTT style annotations to ``video_id -> captions``."""

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
    """Parse ICDAR 2015 localization/transcription ground truth."""

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
    """Validate a synthesized Piper WAV file."""

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
    """Return default Piper voice entries from the TTS config."""

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
    """Validate Piper voice config and optionally synthesize sample WAV files."""

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
    """Run Piper CLI for one prompt."""

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


def augment_preview(
    input_dir: Path,
    output_dir: Path,
    limit: int = 8,
    dry_run: bool = True,
) -> int:
    """Create or preview simple augmentation samples for image datasets."""

    image_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:limit]
    if dry_run:
        return len(image_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            resized_image = rgb_image.resize((640, 640))
            flipped = resized_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped.save(output_dir / image_path.name)
    return len(image_paths)


def read_classes(path: Path) -> list[str]:
    """Read class names from a text file."""

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
    augment.add_argument("--limit", type=int, default=8)
    augment.add_argument("--execute", action="store_true")

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
        )
        print(f"augmented_or_previewed={count} dry_run={not args.execute}")
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
    """Print a validation report for CLI usage."""

    print(f"checked_files={report.checked_files} valid={report.valid}")
    for issue in report.issues:
        print(f"{issue.level}: {issue.path}: {issue.message}")


def print_merge_report(report: MergeReport) -> None:
    """Print a merge report for CLI usage."""

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
