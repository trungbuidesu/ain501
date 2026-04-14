# -*- coding: utf-8 -*-
"""Kiểm thử các tiện ích chuẩn bị dataset Phase 0."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from src.training.data.ocr.textocr import (
    parse_textocr_annotations as parse_textocr_annotations_from_module,
)
from src.training.scripts.data_utils import (
    add_frame_sequence_columns,
    augment_preview,
    augmentation_preset,
    build_action_manifest,
    build_image_augmentation,
    build_scene_subset_manifest,
    build_scene_subset_manifest_from_places_filelist,
    coco_annotations_to_yolo_lines,
    group_aware_split,
    load_yaml,
    main,
    merge_yolo_datasets,
    normalize_coco_bbox,
    normalize_custom_yolo_export,
    parse_icdar_gt,
    parse_msvd_captions,
    parse_places_categories,
    parse_textocr_annotations,
    sample_frame_indices,
    stratified_split,
    validate_tts_config,
    validate_video_manifest,
    validate_video_open,
    validate_wav,
    validate_yolo_dataset,
    verify_dataset_paths,
    write_classes_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write_tiny_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(path)


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.write_text(
        "\n".join(
            [
                ",".join(fieldnames),
                *(",".join(row.get(field, "") for field in fieldnames) for row in rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_coco_to_yolo_keeps_dog_class() -> None:
    """Đảm bảo converter COCO -> YOLO giữ class `dog`."""
    coco = json.loads((FIXTURES / "coco_tiny.json").read_text(encoding="utf-8"))
    class_names = ["person", "dog"]

    labels = coco_annotations_to_yolo_lines(coco, class_names)

    assert labels["000000000001.jpg"] == [
        "0 0.200000 0.200000 0.200000 0.200000",
        "1 0.050000 0.100000 0.100000 0.200000",
    ]


def test_normalize_coco_bbox_clips_and_rejects_invalid() -> None:
    """Đảm bảo bbox COCO được clip đúng và bbox không hợp lệ bị loại."""
    assert normalize_coco_bbox([-10, -10, 20, 20], 100, 100) == pytest.approx(
        (0.05, 0.05, 0.1, 0.1)
    )
    assert normalize_coco_bbox([10, 10, 0, 20], 100, 100) is None


def test_validate_yolo_dataset_reports_bad_labels(tmp_path: Path) -> None:
    """Đảm bảo validator YOLO báo lỗi với class id ngoài danh sách."""
    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("3 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    report = validate_yolo_dataset(root, ["person", "dog"])

    assert not report.valid
    assert any("unknown class id" in issue.message for issue in report.issues)


def test_merge_yolo_dry_run_previews_remap_without_writing(tmp_path: Path) -> None:
    """Đảm bảo merge YOLO dry-run preview remap nhưng không ghi output."""
    source = tmp_path / "source"
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (source / "classes.txt").write_text("dog\n", encoding="utf-8")
    output = tmp_path / "merged"

    report = merge_yolo_datasets(
        sources=[source],
        output_root=output,
        canonical_names=["person", "dog"],
        dry_run=True,
    )

    assert report.valid
    assert report.scanned_labels == 1
    assert report.class_remap[str(source)] == {0: 1}
    assert not output.exists()


def test_normalize_custom_yolo_executes_roboflow_layout(tmp_path: Path) -> None:
    """Đảm bảo normalizer remap export YOLO dạng Roboflow/Ultralytics."""
    export = tmp_path / "roboflow"
    output = tmp_path / "canonical"
    (export / "train" / "labels").mkdir(parents=True)
    _write_tiny_image(export / "train" / "images" / "sample.jpg")
    (export / "train" / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )
    (export / "data.yaml").write_text(
        "names:\n  - dog\n",
        encoding="utf-8",
    )

    report = normalize_custom_yolo_export(
        export_root=export,
        output_root=output,
        canonical_names=["person", "dog"],
        source_format="roboflow_yolo",
        dry_run=False,
    )

    assert report.valid
    assert report.scanned_labels == 1
    assert report.copied_files == 2
    assert (output / "classes.txt").read_text(encoding="utf-8") == "person\ndog\n"
    assert (output / "labels" / "train" / "sample.txt").read_text(
        encoding="utf-8"
    ) == "1 0.5 0.5 0.25 0.25\n"
    assert (output / "images" / "train" / "sample.jpg").exists()


def test_normalize_custom_yolo_dry_run_cvat_layout(tmp_path: Path) -> None:
    """Đảm bảo normalizer đọc CVAT YOLO flat layout mà không ghi khi dry-run."""
    export = tmp_path / "cvat"
    output = tmp_path / "canonical"
    (export / "obj_train_data").mkdir(parents=True)
    _write_tiny_image(export / "obj_train_data" / "sample.jpg")
    (export / "obj_train_data" / "sample.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )
    (export / "obj.names").write_text("dog\n", encoding="utf-8")

    report = normalize_custom_yolo_export(
        export_root=export,
        output_root=output,
        canonical_names=["person", "dog"],
        source_format="cvat_yolo",
        dry_run=True,
    )

    assert report.valid
    assert report.scanned_labels == 1
    assert report.copied_files == 0
    assert not output.exists()


def test_split_helpers_keep_expected_invariants() -> None:
    """Đảm bảo split helper giữ invariant về label và group."""
    rows = [{"path": f"a{i}", "label": "a", "group": f"ga{i}"} for i in range(10)] + [
        {"path": f"b{i}", "label": "b", "group": f"gb{i}"} for i in range(10)
    ]

    stratified = stratified_split(rows, label_key="label")
    assert {row["label"] for row in stratified["train"]} == {"a", "b"}

    grouped = group_aware_split(rows, group_key="group")
    split_by_group = {
        row["group"]: split_name
        for split_name, split_rows in grouped.items()
        for row in split_rows
    }
    assert len(split_by_group) == len({row["group"] for row in rows})


def test_public_dataset_parsers() -> None:
    """Đảm bảo parser fixture cho MSVD, Places365, ICDAR và TextOCR hoạt động."""
    captions = parse_msvd_captions(FIXTURES / "msvd_tiny.json")
    assert captions["video0"] == [
        "a person walks on the sidewalk",
        "a dog waits near a person",
    ]
    assert captions["video1"] == ["a bus approaches the stop"]

    places = parse_places_categories(FIXTURES / "places_categories_tiny.txt")
    assert places["a/crosswalk"] == 0

    icdar = parse_icdar_gt(FIXTURES / "icdar_tiny.txt")
    assert icdar[0].text == "STORE"
    assert icdar[1].ignored

    textocr = parse_textocr_annotations(FIXTURES / "textocr_tiny.json")
    assert textocr[0].image_file == "train_val_images/img0.jpg"
    assert textocr[0].bbox == (10.0, 20.0, 40.0, 12.0)
    assert textocr[0].text == "STORE"
    assert textocr[1].ignored

    module_textocr = parse_textocr_annotations_from_module(
        FIXTURES / "textocr_tiny.json"
    )
    assert module_textocr == textocr


def test_tts_config_and_wav_validation(tmp_path: Path) -> None:
    """Đảm bảo config TTS và WAV sample tối thiểu được validate đúng."""
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    wav_path = tmp_path / "voice.wav"
    model_path.write_bytes(b"fake")
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")
    tts_config = tmp_path / "tts.yaml"
    tts_config.write_text(
        "\n".join(
            [
                "paths:",
                f"  validation_output: {tmp_path.as_posix()}",
                "voices:",
                "  defaults:",
                "    en:",
                "      id: en_US-test",
                f"      model_path: {model_path.as_posix()}",
                f"      config_path: {config_path.as_posix()}",
                "      expected_sample_rate_hz: 22050",
                "      prompt: test",
            ]
        ),
        encoding="utf-8",
    )
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 10)

    assert validate_tts_config(tts_config, strict_files=True).valid
    assert validate_wav(wav_path, expected_sample_rate_hz=22050).valid


def test_cli_smoke_dry_run() -> None:
    """Đảm bảo các lệnh CLI dry-run chính chạy thành công."""
    assert main(["download-plan"]) == 0
    assert main(["validate-tts", "--dry-run"]) == 0


def test_build_action_manifest_requires_public_bootstrap_list(tmp_path: Path) -> None:
    """Đảm bảo CLI báo lỗi khi classes.public_bootstrap không phải list."""
    config = tmp_path / "action_invalid.yaml"
    config.write_text(
        "\n".join(
            [
                "base_dataset:",
                "  videos_root: /tmp/videos",
                "  split_root: /tmp/splits",
                "classes:",
                "  public_bootstrap: not-a-list",
                "output:",
                "  manifest: /tmp/action_manifest.csv",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Action config classes.public_bootstrap must be a list",
    ):
        main(["build-action-manifest", "--config", str(config)])


def test_build_scene_subset_requires_places_subset_list(tmp_path: Path) -> None:
    """Đảm bảo CLI báo lỗi khi classes.places_subset không phải list."""
    config = tmp_path / "scene_invalid.yaml"
    config.write_text(
        "\n".join(
            [
                "base_dataset:",
                "  root: /tmp/images",
                "  categories_file: /tmp/categories.txt",
                "classes:",
                "  places_subset: not-a-list",
                "balancing:",
                "  max_images_per_class: 100",
                "output:",
                "  manifest: /tmp/scene_manifest.csv",
                "  subset_root: /tmp/subset",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Scene config classes.places_subset must be a list",
    ):
        main(["build-scene-subset", "--config", str(config)])


def test_config_contains_accessibility_requirements() -> None:
    """Đảm bảo config chứa các yêu cầu accessibility quan trọng."""
    detection = load_yaml(Path("configs/datasets/object_detection_accessibility.yaml"))
    action = load_yaml(Path("configs/datasets/action_accessibility.yaml"))
    assert "dog" in detection["classes"]["coco_subset"]
    assert "Phase 9" in action["phase_notes"]["approaching"]


def test_ucf101_action_manifest_uses_official_split_lists(tmp_path: Path) -> None:
    """Đảm bảo action manifest đọc official UCF-101 train/test split lists."""
    videos_root = tmp_path / "UCF-101"
    split_root = tmp_path / "ucfTrainTestlist"
    (videos_root / "Biking").mkdir(parents=True)
    (videos_root / "WalkingWithDog").mkdir()
    split_root.mkdir()
    (videos_root / "Biking" / "v_Biking_g01_c01.avi").write_bytes(b"fake")
    (videos_root / "WalkingWithDog" / "v_WalkingWithDog_g01_c01.avi").write_bytes(
        b"fake"
    )
    (split_root / "trainlist01.txt").write_text(
        "Biking/v_Biking_g01_c01.avi 10\n",
        encoding="utf-8",
    )
    (split_root / "testlist01.txt").write_text(
        "WalkingWithDog/v_WalkingWithDog_g01_c01.avi\n",
        encoding="utf-8",
    )

    rows = build_action_manifest(
        videos_root=videos_root,
        class_names=["Biking", "WalkingWithDog"],
        dataset_name="ucf101",
        split_root=split_root,
    )

    assert [row["relative_path"] for row in rows] == [
        "Biking/v_Biking_g01_c01.avi",
        "WalkingWithDog/v_WalkingWithDog_g01_c01.avi",
    ]
    assert {row["split"] for row in rows} == {"train", "test"}
    assert validate_video_manifest(rows, ["Biking", "WalkingWithDog"]).valid


def test_frame_sequence_sampling_and_manifest_validation(tmp_path: Path) -> None:
    """Đảm bảo sampler giữ stride và validator bắt group leak giữa split."""
    video = tmp_path / "walk.avi"
    video.write_bytes(b"fake")
    rows = [
        {
            "path": str(video),
            "relative_path": "walk/walk.avi",
            "label": "walk",
            "group": "shared",
            "split": "train",
        }
    ]

    assert sample_frame_indices(33, sequence_length=16, frame_stride=2) == tuple(
        range(1, 32, 2)
    )
    assert sample_frame_indices(30, sequence_length=16, frame_stride=2) == ()

    sequence_rows = add_frame_sequence_columns(
        rows,
        frame_counts={str(video): 33},
        sequence_length=16,
        frame_stride=2,
    )
    assert sequence_rows[0]["frame_indices"] == " ".join(
        str(index) for index in range(1, 32, 2)
    )

    leaking_rows = [
        *rows,
        {
            **rows[0],
            "path": str(video),
            "relative_path": "walk/walk-copy.avi",
            "split": "test",
        },
    ]
    report = validate_video_manifest(leaking_rows, ["walk"])
    assert not report.valid
    assert any("multiple splits" in issue.message for issue in report.issues)

    bad_rows = [
        {
            **rows[0],
            "path": str(tmp_path / "missing.avi"),
        },
        {
            **rows[0],
            "label": "run",
        },
    ]
    bad_report = validate_video_manifest(bad_rows, ["walk"])
    assert not bad_report.valid
    assert any("missing video file" in issue.message for issue in bad_report.issues)
    assert any("unknown class" in issue.message for issue in bad_report.issues)


def test_validate_video_manifest_cli_and_open_check(tmp_path: Path) -> None:
    """Đảm bảo CLI validate manifest và optional OpenCV check báo lỗi file giả."""
    video = tmp_path / "walk.avi"
    video.write_bytes(b"fake")
    classes = tmp_path / "classes.txt"
    classes.write_text("walk\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [
            {
                "path": str(video),
                "relative_path": "walk/walk.avi",
                "label": "walk",
                "group": "walk_g01",
                "split": "train",
            }
        ],
    )

    assert (
        main(
            [
                "validate-video-manifest",
                "--manifest-csv",
                str(manifest),
                "--classes",
                str(classes),
            ]
        )
        == 0
    )
    assert validate_video_open(video) is not None
    assert (
        main(
            [
                "validate-video-manifest",
                "--manifest-csv",
                str(manifest),
                "--classes",
                str(classes),
                "--check-video-open",
            ]
        )
        == 1
    )


def test_augmentation_presets_and_dry_run_do_not_write(tmp_path: Path) -> None:
    """Đảm bảo preset augmentation tồn tại và dry-run không ghi output."""
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "augmented"
    input_dir.mkdir()
    Image.new("RGB", (16, 16), color="white").save(input_dir / "sample.jpg")

    for task in ("detection", "scene", "ocr"):
        preset = augmentation_preset(task, size=32)
        transform = build_image_augmentation(task, size=32, normalize=False)
        assert preset.task == task
        assert callable(transform)

    count = augment_preview(
        input_dir=input_dir,
        output_dir=output_dir,
        task="scene",
        size=32,
        dry_run=True,
    )
    assert count == 1
    assert not output_dir.exists()


def test_src_training_package_imports_outside_repo_root(tmp_path: Path) -> None:
    """Đảm bảo package editable expose `src.training.scripts` ngoài repo root."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.training.scripts.data_utils as d; print(d.__name__)",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "src.training.scripts.data_utils" in result.stdout


def test_write_classes_and_verify_dataset_paths(tmp_path: Path) -> None:
    """Đảm bảo class writer dry-run an toàn và path verifier báo thiếu path."""
    config_path = tmp_path / "dataset.yaml"
    data_root = tmp_path / "external" / "demo"
    config_path.write_text(
        "\n".join(
            [
                "name: demo",
                "base_dataset:",
                f"  root: {data_root.as_posix()}",
                "classes:",
                "  coco_subset:",
                "    - person",
                "    - dog",
                "  custom_only:",
                "    - door",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "classes.txt"

    class_names = write_classes_file(config_path, output_path, dry_run=True)

    assert class_names == ["person", "dog"]
    assert not output_path.exists()
    report = verify_dataset_paths([config_path])
    assert not report.valid
    assert any(str(data_root) in issue.message for issue in report.issues)


def test_scene_subset_manifest_balances_by_class(tmp_path: Path) -> None:
    """Đảm bảo scene subset chọn tối đa số ảnh cấu hình cho mỗi class."""
    images_root = tmp_path / "places"
    (images_root / "a" / "crosswalk").mkdir(parents=True)
    (images_root / "restaurant").mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8), color="white").save(
            images_root / "a" / "crosswalk" / f"crosswalk_{index}.jpg"
        )
    Image.new("RGB", (8, 8), color="white").save(
        images_root / "restaurant" / "restaurant_0.jpg"
    )

    rows = build_scene_subset_manifest(
        images_root,
        ["crosswalk", "restaurant"],
        max_images_per_class=2,
    )

    assert [row["label"] for row in rows].count("crosswalk") == 2
    assert [row["label"] for row in rows].count("restaurant") == 1
    assert all(Path(row["path"]).exists() for row in rows)


def test_scene_subset_manifest_from_places_filelist(tmp_path: Path) -> None:
    """Đảm bảo scene subset đọc được official Places365 val filelist."""
    images_root = tmp_path / "places365"
    val_root = images_root / "val_256"
    val_root.mkdir(parents=True)
    for name in [
        "Places365_val_00000001.jpg",
        "Places365_val_00000002.jpg",
        "Places365_val_00000003.jpg",
    ]:
        Image.new("RGB", (8, 8), color="white").save(val_root / name)
    categories_file = images_root / "categories_places365.txt"
    categories_file.write_text(
        "/c/crosswalk 0\n/r/restaurant 1\n/s/street 2\n",
        encoding="utf-8",
    )
    filelist_path = images_root / "places365_val.txt"
    filelist_path.write_text(
        "\n".join(
            [
                "/Places365_val_00000001.jpg 0",
                "/Places365_val_00000002.jpg 0",
                "/Places365_val_00000003.jpg 1",
            ]
        ),
        encoding="utf-8",
    )

    rows = build_scene_subset_manifest_from_places_filelist(
        images_root=images_root,
        filelist_path=filelist_path,
        categories_file=categories_file,
        class_names=["crosswalk", "restaurant"],
        max_images_per_class=1,
    )

    assert [row["label"] for row in rows] == ["crosswalk", "restaurant"]
    assert all(Path(row["path"]).exists() for row in rows)


def test_scene_subset_filelist_requires_full_match_for_nested_class(
    tmp_path: Path,
) -> None:
    """Đảm bảo class nested không match nhầm mọi category cùng segment cuối."""
    images_root = tmp_path / "places365"
    val_root = images_root / "val_256"
    val_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(
        val_root / "Places365_val_00000001.jpg"
    )
    categories_file = images_root / "categories_places365.txt"
    categories_file.write_text(
        "/r/restaurant/indoor 0\n/g/general_store/indoor 1\n",
        encoding="utf-8",
    )
    filelist_path = images_root / "places365_val.txt"
    filelist_path.write_text(
        "/Places365_val_00000001.jpg 0\n",
        encoding="utf-8",
    )

    rows = build_scene_subset_manifest_from_places_filelist(
        images_root=images_root,
        filelist_path=filelist_path,
        categories_file=categories_file,
        class_names=["general_store/indoor"],
        max_images_per_class=1,
    )

    assert rows == []
