# -*- coding: utf-8 -*-
"""Kiểm thử các tiện ích chuẩn bị dataset Phase 0."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from PIL import Image

from training.scripts.data_utils import (
    coco_annotations_to_yolo_lines,
    group_aware_split,
    load_yaml,
    main,
    merge_yolo_datasets,
    normalize_coco_bbox,
    parse_icdar_gt,
    parse_msr_vtt_captions,
    parse_places_categories,
    stratified_split,
    validate_tts_config,
    validate_wav,
    validate_yolo_dataset,
)

FIXTURES = Path(__file__).parent / "fixtures"


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
    """Đảm bảo parser fixture cho MSR-VTT, Places365 và ICDAR hoạt động."""
    captions = parse_msr_vtt_captions(FIXTURES / "msr_vtt_tiny.json")
    assert captions["video0"] == [
        "a person walks on the sidewalk",
        "a dog waits near a person",
    ]

    places = parse_places_categories(FIXTURES / "places_categories_tiny.txt")
    assert places["a/crosswalk"] == 0

    icdar = parse_icdar_gt(FIXTURES / "icdar_tiny.txt")
    assert icdar[0].text == "STORE"
    assert icdar[1].ignored


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


def test_config_contains_accessibility_requirements() -> None:
    """Đảm bảo config chứa các yêu cầu accessibility quan trọng."""
    detection = load_yaml(Path("configs/datasets/object_detection_accessibility.yaml"))
    action = load_yaml(Path("configs/datasets/action_accessibility.yaml"))
    assert "dog" in detection["classes"]["coco_subset"]
    assert "Phase 9" in action["phase_notes"]["approaching"]
