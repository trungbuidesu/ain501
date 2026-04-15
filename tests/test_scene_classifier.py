# -*- coding: utf-8 -*-
"""Smoke tests for scene classifier pipeline."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.training.models.vision import MobileNetSceneHead, TinySceneCNN
from src.training.scripts import scene_classifier


def test_scene_classifier_model_shapes() -> None:
    """Tiny CNN and MobileNet head both return 5-class logits."""

    tiny = TinySceneCNN(num_classes=5).eval()
    mobile = MobileNetSceneHead(num_classes=5, weights="none").eval()
    sample = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        tiny_logits = tiny(sample)
        mobile_logits = mobile(sample)
    assert tuple(tiny_logits.shape) == (2, 5)
    assert tuple(mobile_logits.shape) == (2, 5)


def test_prepare_manifest_train_qat_export_and_benchmark(tmp_path: Path) -> None:
    """Run end-to-end smoke path on synthetic data."""

    synthetic = _build_synthetic_sources(tmp_path)
    prepared_manifest = tmp_path / "prepared_manifest.csv"
    checkpoint = tmp_path / "models" / "smoke" / "best.pt"
    qat_checkpoint = tmp_path / "models" / "smoke" / "qat.pt"
    onnx_fp32 = tmp_path / "models" / "scene_classifier_fp32.onnx"
    onnx_int8 = tmp_path / "models" / "scene_classifier_int8.onnx"
    benchmark_report = tmp_path / "reports" / "benchmark.json"

    prepare_exit = scene_classifier.main(
        [
            "prepare-manifest",
            "--scene-manifest",
            str(synthetic["scene_manifest"]),
            "--ucf-frames-manifest",
            str(synthetic["ucf_manifest"]),
            "--textocr-train-ann",
            str(synthetic["textocr_train"]),
            "--textocr-val-ann",
            str(synthetic["textocr_val"]),
            "--textocr-images-root",
            str(synthetic["textocr_images"]),
            "--face-roots",
            str(synthetic["face_root"]),
            "--min-per-class",
            "2",
            "--max-per-class",
            "2",
            "--output-csv",
            str(prepared_manifest),
            "--execute",
        ]
    )
    assert prepare_exit == 0
    rows = _read_rows(prepared_manifest)
    assert len(rows) == 10
    assert {row["label"] for row in rows} == {
        "static",
        "motion",
        "text_present",
        "face_detected",
        "scene_change",
    }

    train_exit = scene_classifier.main(
        [
            "train",
            "--manifest-csv",
            str(prepared_manifest),
            "--variant",
            "tiny_cnn",
            "--weights",
            "none",
            "--run-name",
            "smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "96",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(tmp_path / "models"),
            "--log-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert train_exit == 0
    assert checkpoint.exists()

    evaluate_exit = scene_classifier.main(
        [
            "evaluate",
            "--manifest-csv",
            str(prepared_manifest),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert evaluate_exit == 0

    qat_exit = scene_classifier.main(
        [
            "qat",
            "--manifest-csv",
            str(prepared_manifest),
            "--checkpoint",
            str(checkpoint),
            "--output-checkpoint",
            str(qat_checkpoint),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert qat_exit == 0
    assert qat_checkpoint.exists()

    export_exit = scene_classifier.main(
        [
            "export-onnx",
            "--checkpoint",
            str(checkpoint),
            "--output-fp32",
            str(onnx_fp32),
            "--output-int8",
            str(onnx_int8),
            "--verify-parity",
        ]
    )
    assert export_exit == 0
    assert onnx_fp32.exists()
    assert onnx_int8.exists()

    benchmark_exit = scene_classifier.main(
        [
            "benchmark",
            "--runtime",
            "onnxruntime",
            "--onnx-path",
            str(onnx_int8),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--image-size",
            "96",
            "--output-json",
            str(benchmark_report),
        ]
    )
    assert benchmark_exit == 0
    report = json.loads(benchmark_report.read_text(encoding="utf-8"))
    assert report["runtime"] == "onnxruntime"
    assert report["latency_ms_p50"] > 0
    assert "ram_mb_delta" in report


def _build_synthetic_sources(tmp_path: Path) -> dict[str, Path]:
    scene_images = tmp_path / "scene_images"
    scene_images.mkdir(parents=True)
    scene_manifest = tmp_path / "scene_manifest.csv"
    scene_rows: list[dict[str, str]] = []
    for index in range(4):
        image_path = scene_images / f"static_{index}.jpg"
        Image.new("RGB", (80, 80), color=(10 + index, 20, 30)).save(image_path)
        scene_rows.append({"path": str(image_path), "label": "street"})
    _write_rows(scene_manifest, scene_rows, ["path", "label"])

    ucf_root = tmp_path / "ucf_frames"
    ucf_root.mkdir(parents=True)
    ucf_manifest = tmp_path / "ucf_manifest.csv"
    ucf_rows: list[dict[str, str]] = []
    for clip_index in range(4):
        frame_dir = ucf_root / f"clip_{clip_index}"
        frame_dir.mkdir(parents=True)
        Image.new("RGB", (80, 80), color=(clip_index * 10, 10, 10)).save(
            frame_dir / "frame_00000.jpg"
        )
        Image.new("RGB", (80, 80), color=(255 - clip_index * 10, 230, 200)).save(
            frame_dir / "frame_00001.jpg"
        )
        ucf_rows.append({"frame_dir": str(frame_dir)})
    _write_rows(ucf_manifest, ucf_rows, ["frame_dir"])

    textocr_images = tmp_path / "textocr_images"
    textocr_images.mkdir(parents=True)
    textocr_train = tmp_path / "textocr_train.json"
    textocr_val = tmp_path / "textocr_val.json"
    train_json = _build_textocr_json(textocr_images, prefix="train")
    val_json = _build_textocr_json(textocr_images, prefix="val")
    textocr_train.write_text(json.dumps(train_json), encoding="utf-8")
    textocr_val.write_text(json.dumps(val_json), encoding="utf-8")

    face_root = tmp_path / "face_dataset"
    face_root.mkdir(parents=True)
    for index in range(4):
        Image.new("RGB", (64, 64), color=(120, index * 20, 80)).save(
            face_root / f"face_{index}.jpg"
        )

    return {
        "scene_manifest": scene_manifest,
        "ucf_manifest": ucf_manifest,
        "textocr_train": textocr_train,
        "textocr_val": textocr_val,
        "textocr_images": textocr_images,
        "face_root": face_root,
    }


def _build_textocr_json(images_root: Path, prefix: str) -> dict[str, object]:
    imgs: dict[str, dict[str, object]] = {}
    anns: dict[str, dict[str, object]] = {}
    for index in range(2):
        file_name = f"{prefix}_{index}.jpg"
        image_path = images_root / file_name
        pixels = np.full((70, 70, 3), fill_value=30 + index * 20, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(image_path)
        image_id = str(index)
        imgs[image_id] = {"id": index, "file_name": file_name}
        anns[str(index)] = {
            "image_id": image_id,
            "utf8_string": "text",
            "bbox": [5, 5, 20, 15],
            "points": [[5, 5], [25, 5], [25, 20], [5, 20]],
            "legibility": "legible",
        }
    return {"imgs": imgs, "anns": anns}


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
