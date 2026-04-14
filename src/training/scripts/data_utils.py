# -*- coding: utf-8 -*-
"""Tiện ích chuẩn bị dataset cho Phase 0 của accessibility assistant.

Entrypoint CLI::

    python -m src.training.scripts.data_utils <subcommand> [--help]

Các subcommand chính: ``download-plan``, ``verify-dataset-paths``,
``write-classes``, ``split``, ``validate`` (YOLO), ``convert`` (COCO->YOLO),
``merge-yolo``, ``normalize-custom-yolo``, ``augment-preview``,
``build-action-manifest``, ``validate-video-manifest``, ``build-scene-subset``,
``validate-tts``. Thao tác ghi file hoặc tải dữ liệu lớn thường cần cờ
``--execute`` sau khi đã xem kế hoạch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.training.data import (
    DEFAULT_DATASET_CONFIG_DIR,
    AugmentationPreset,
    IcdarTextBox,
    MergeReport,
    NormalizeYoloReport,
    TextOcrTextBox,
    ValidationIssue,
    ValidationReport,
    add_frame_sequence_columns,
    augment_preview,
    augmentation_preset,
    build_action_manifest,
    build_image_augmentation,
    build_scene_subset_manifest,
    build_scene_subset_manifest_from_places_filelist,
    canonical_class_names,
    coco_annotations_to_yolo_lines,
    copy_scene_subset,
    dataset_config_paths,
    download_plan_lines,
    group_aware_split,
    load_yaml,
    merge_yolo_datasets,
    normalize_coco_bbox,
    normalize_custom_yolo_export,
    parse_icdar_gt,
    parse_msvd_captions,
    parse_places_categories,
    parse_textocr_annotations,
    read_action_classes,
    read_action_split_assignments,
    read_classes,
    read_places_categories,
    read_ucf101_split_assignments,
    read_yolo_names,
    remap_yolo_label_text,
    sample_frame_indices,
    stratified_split,
    synthesize_with_piper,
    tts_voice_entries,
    validate_tts_config,
    validate_video_manifest,
    validate_video_open,
    validate_wav,
    validate_yolo_dataset,
    validate_yolo_label_line,
    verify_dataset_paths,
    write_classes_file,
    write_manifest_csv,
    write_split_csvs,
    write_yolo_from_coco,
)
from src.training.data.scene_classification.places365 import (
    _find_default_places_filelist,
)

__all__ = [
    "DEFAULT_DATASET_CONFIG_DIR",
    "AugmentationPreset",
    "IcdarTextBox",
    "MergeReport",
    "NormalizeYoloReport",
    "TextOcrTextBox",
    "ValidationIssue",
    "ValidationReport",
    "add_frame_sequence_columns",
    "augment_preview",
    "augmentation_preset",
    "build_action_manifest",
    "build_arg_parser",
    "build_image_augmentation",
    "build_scene_subset_manifest",
    "build_scene_subset_manifest_from_places_filelist",
    "canonical_class_names",
    "coco_annotations_to_yolo_lines",
    "copy_scene_subset",
    "dataset_config_paths",
    "download_plan_lines",
    "group_aware_split",
    "load_yaml",
    "main",
    "merge_yolo_datasets",
    "normalize_coco_bbox",
    "normalize_custom_yolo_export",
    "parse_icdar_gt",
    "parse_msvd_captions",
    "parse_places_categories",
    "parse_textocr_annotations",
    "read_action_classes",
    "print_merge_report",
    "print_normalize_yolo_report",
    "print_report",
    "read_action_split_assignments",
    "read_classes",
    "read_places_categories",
    "read_ucf101_split_assignments",
    "read_yolo_names",
    "remap_yolo_label_text",
    "sample_frame_indices",
    "stratified_split",
    "synthesize_with_piper",
    "tts_voice_entries",
    "validate_tts_config",
    "validate_video_manifest",
    "validate_video_open",
    "validate_wav",
    "validate_yolo_dataset",
    "validate_yolo_label_line",
    "verify_dataset_paths",
    "write_classes_file",
    "write_manifest_csv",
    "write_split_csvs",
    "write_yolo_from_coco",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo parser dòng lệnh cho các tiện ích data pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-plan")
    download.add_argument("--config-dir", type=Path, default=DEFAULT_DATASET_CONFIG_DIR)
    download.add_argument("--execute", action="store_true")

    verify_paths = subparsers.add_parser("verify-dataset-paths")
    verify_paths.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_DIR,
    )
    verify_paths.add_argument("--tasks", nargs="*")
    verify_paths.add_argument("--include-outputs", action="store_true")
    verify_paths.add_argument("--skip-downloads", action="store_true")

    write_classes = subparsers.add_parser("write-classes")
    write_classes.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/object_detection_accessibility.yaml"),
    )
    write_classes.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/object_detection_accessibility/classes.txt"),
    )
    write_classes.add_argument("--class-key", default="coco_subset")
    write_classes.add_argument("--include-custom", action="store_true")
    write_classes.add_argument("--execute", action="store_true")

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
    merge_mode = merge.add_mutually_exclusive_group()
    merge_mode.add_argument("--dry-run", action="store_true")
    merge_mode.add_argument("--execute", action="store_true")

    normalize_custom_yolo = subparsers.add_parser("normalize-custom-yolo")
    normalize_custom_yolo.add_argument("--export-root", type=Path, required=True)
    normalize_custom_yolo.add_argument("--output-root", type=Path, required=True)
    normalize_custom_yolo.add_argument("--classes", type=Path, required=True)
    normalize_custom_yolo.add_argument(
        "--source-format",
        choices=["roboflow_yolo", "cvat_yolo", "ultralytics_yolo"],
        default="roboflow_yolo",
    )
    normalize_custom_yolo.add_argument("--source-classes", type=Path)
    normalize_custom_yolo_mode = normalize_custom_yolo.add_mutually_exclusive_group()
    normalize_custom_yolo_mode.add_argument("--dry-run", action="store_true")
    normalize_custom_yolo_mode.add_argument("--execute", action="store_true")

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

    validate_video = subparsers.add_parser("validate-video-manifest")
    validate_video.add_argument("--manifest-csv", type=Path, required=True)
    validate_video.add_argument("--classes", type=Path)
    validate_video.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/action_accessibility.yaml"),
    )
    validate_video.add_argument("--class-key", default="public_bootstrap")
    validate_video.add_argument("--check-video-open", action="store_true")
    validate_video.add_argument("--min-frames", type=int, default=1)

    scene_subset = subparsers.add_parser("build-scene-subset")
    scene_subset.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/scene_accessibility.yaml"),
    )
    scene_subset.add_argument("--images-root", type=Path)
    scene_subset.add_argument("--file-list", type=Path)
    scene_subset.add_argument("--categories-file", type=Path)
    scene_subset.add_argument("--output-csv", type=Path)
    scene_subset.add_argument("--subset-root", type=Path)
    scene_subset.add_argument("--max-images-per-class", type=int)
    scene_subset.add_argument("--copy-images", action="store_true")
    scene_subset.add_argument("--execute", action="store_true")

    validate_tts_parser = subparsers.add_parser("validate-tts")
    validate_tts_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/tts_piper_accessibility.yaml"),
    )
    validate_tts_parser.add_argument("--execute", action="store_true")
    validate_tts_parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_config_section(
    config: dict[str, Any],
    section: str,
    *,
    context_name: str,
) -> dict[str, Any]:
    """Đọc một section dạng mapping từ config và raise lỗi nếu thiếu/sai kiểu."""
    value = config.get(section, {})
    if not isinstance(value, dict):
        raise ValueError(f"{context_name} missing {section}")
    return value


def _read_config_list(
    section: dict[str, Any],
    key: str,
    *,
    error_message: str,
) -> list[Any]:
    """Đọc một field list từ config section và validate kiểu dữ liệu."""
    value = section.get(key, [])
    if not isinstance(value, list):
        raise ValueError(error_message)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI data pipeline và trả về mã thoát."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "download-plan":
        if args.execute:
            print("Execute mode is reserved for explicit dataset download wrappers.")
        print("\n".join(download_plan_lines(dataset_config_paths(args.config_dir))))
        return 0

    if args.command == "verify-dataset-paths":
        report = verify_dataset_paths(
            dataset_config_paths(args.config_dir),
            tasks=args.tasks,
            include_outputs=args.include_outputs,
            include_downloads=not args.skip_downloads,
        )
        print_report(report)
        return 0 if report.valid else 1

    if args.command == "write-classes":
        class_names = write_classes_file(
            config_path=args.config,
            output_path=args.output,
            class_key=args.class_key,
            include_custom=args.include_custom,
            dry_run=not args.execute,
        )
        print(
            " ".join(
                [
                    f"classes={len(class_names)}",
                    f"output={args.output}",
                    f"dry_run={not args.execute}",
                ]
            )
        )
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
            dry_run=not args.execute,
        )
        print_merge_report(merge_report)
        return 0 if merge_report.valid else 1

    if args.command == "normalize-custom-yolo":
        normalize_report = normalize_custom_yolo_export(
            export_root=args.export_root,
            output_root=args.output_root,
            canonical_names=read_classes(args.classes),
            source_format=args.source_format,
            classes_path=args.source_classes,
            dry_run=not args.execute,
        )
        print_normalize_yolo_report(normalize_report)
        return 0 if normalize_report.valid else 1

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
        base_dataset = _read_config_section(
            config,
            "base_dataset",
            context_name="Action config",
        )
        classes = _read_config_section(config, "classes", context_name="Action config")
        output = _read_config_section(config, "output", context_name="Action config")
        public_classes = _read_config_list(
            classes,
            "public_bootstrap",
            error_message="Action config classes.public_bootstrap must be a list",
        )
        videos_root = args.videos_root or Path(str(base_dataset["videos_root"]))
        split_root = args.split_root or Path(str(base_dataset["split_root"]))
        output_csv = args.output_csv or Path(str(output["manifest"]))
        rows = build_action_manifest(
            videos_root=videos_root,
            class_names=[str(name) for name in public_classes],
            dataset_name=str(base_dataset.get("name", "class_folders")),
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

    if args.command == "validate-video-manifest":
        with args.manifest_csv.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        class_names = (
            read_classes(args.classes)
            if args.classes
            else read_action_classes(args.config, args.class_key)
        )
        video_report = validate_video_manifest(
            rows,
            class_names,
            check_video_open=args.check_video_open,
            min_frames=args.min_frames,
        )
        print_report(video_report)
        return 0 if video_report.valid else 1

    if args.command == "build-scene-subset":
        config = load_yaml(args.config)
        base_dataset = _read_config_section(
            config,
            "base_dataset",
            context_name="Scene config",
        )
        classes = _read_config_section(config, "classes", context_name="Scene config")
        balancing = _read_config_section(
            config,
            "balancing",
            context_name="Scene config",
        )
        output = _read_config_section(config, "output", context_name="Scene config")
        class_names = _read_config_list(
            classes,
            "places_subset",
            error_message="Scene config classes.places_subset must be a list",
        )
        images_root = args.images_root or Path(str(base_dataset["root"]))
        categories_file = args.categories_file or Path(
            str(base_dataset["categories_file"])
        )
        filelist_path = args.file_list or _find_default_places_filelist(images_root)
        output_csv = args.output_csv or Path(str(output["manifest"]))
        subset_root = args.subset_root or Path(str(output["subset_root"]))
        max_images = args.max_images_per_class or int(
            balancing.get("max_images_per_class", 1000)
        )
        string_class_names = [str(name) for name in class_names]
        if filelist_path and categories_file.exists():
            rows = build_scene_subset_manifest_from_places_filelist(
                images_root=images_root,
                filelist_path=filelist_path,
                categories_file=categories_file,
                class_names=string_class_names,
                max_images_per_class=max_images,
            )
        else:
            rows = build_scene_subset_manifest(
                images_root=images_root,
                class_names=string_class_names,
                max_images_per_class=max_images,
            )
        if args.execute:
            write_manifest_csv(rows, output_csv)
            copied = copy_scene_subset(rows, subset_root) if args.copy_images else 0
            print(
                f"wrote_manifest={output_csv} rows={len(rows)} copied_images={copied}"
            )
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


def print_normalize_yolo_report(report: NormalizeYoloReport) -> None:
    """In report normalize YOLO custom export cho CLI."""

    print(
        " ".join(
            [
                f"dry_run={report.dry_run}",
                f"scanned_labels={report.scanned_labels}",
                f"copied_files={report.copied_files}",
                f"checked_files={report.validation.checked_files}",
                f"valid={report.valid}",
            ]
        )
    )
    for issue in report.validation.issues:
        print(f"{issue.level}: {issue.path}: {issue.message}")


if __name__ == "__main__":
    sys.exit(main())
