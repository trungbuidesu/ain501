# -*- coding: utf-8 -*-
"""Tests for MobileNetV3-Small shared feature extractor utilities."""

import csv
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from src.training.models.vision import (
    MobileNetV3SmallBackbone,
    MobileNetV3SmallWithHead,
    freeze_mobilenetv3_early_layers,
)
from src.training.scripts import mobilenetv3


def test_mobilenetv3_backbone_shapes() -> None:
    """Backbone exposes spatial feature maps and pooled embeddings."""

    model = MobileNetV3SmallBackbone(weights="none").eval()
    inputs = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        feature_map = model.forward_feature_map(inputs)
        embedding = model.forward_embedding(inputs)

    assert tuple(feature_map.shape) == (2, 576, 7, 7)
    assert tuple(embedding.shape) == (2, 576)
    assert tuple(model(inputs).shape) == (2, 576)


def test_mobilenetv3_head_and_freeze_policy() -> None:
    """Temporary head returns class logits and freeze policy keeps later layers open."""

    backbone = MobileNetV3SmallBackbone(weights="none")
    model = MobileNetV3SmallWithHead(backbone, num_classes=3)
    freeze_mobilenetv3_early_layers(model, freeze_until=7)

    early_params = list(model.backbone.features[:7].parameters())
    later_params = list(model.backbone.features[7:].parameters())

    assert early_params
    assert later_params
    assert all(not parameter.requires_grad for parameter in early_params)
    assert any(parameter.requires_grad for parameter in later_params)
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    logits = model(torch.randn(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 3)


def test_fine_tune_smoke_writes_tensorboard(tmp_path: Path) -> None:
    """Fine-tune CLI runs one tiny epoch and writes TensorBoard events."""

    manifest = _write_manifest(tmp_path, image_count=6)
    checkpoint_dir = tmp_path / "models"
    log_dir = tmp_path / "runs"

    exit_code = mobilenetv3.main(
        [
            "fine-tune",
            "--manifest-csv",
            str(manifest),
            "--weights",
            "none",
            "--run-name",
            "smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "64",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--log-dir",
            str(log_dir),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert (checkpoint_dir / "smoke" / "best_backbone.pt").exists()
    assert list((log_dir / "smoke").glob("events.out.tfevents.*"))


def test_export_onnx_embedding_smoke(tmp_path: Path) -> None:
    """ONNX export produces a valid FP32 embedding graph."""

    output = tmp_path / "mobilenetv3_small.onnx"

    exit_code = mobilenetv3.main(
        [
            "export-onnx",
            "--weights",
            "none",
            "--output-kind",
            "embedding",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    model = onnx.load(output)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(
        None,
        {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)},
    )
    assert outputs[0].shape == (1, 576)


def test_benchmark_smoke_outputs_latency_and_ram(tmp_path: Path) -> None:
    """Benchmark CLI emits latency and RAM fields for tiny CPU runs."""

    output = tmp_path / "benchmark.json"

    exit_code = mobilenetv3.main(
        [
            "benchmark",
            "--weights",
            "none",
            "--runtime",
            "pytorch",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--image-size",
            "64",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"] == "pytorch"
    assert report["latency_ms_p50"] > 0
    assert "ram_mb_start" in report
    assert "ram_mb_end" in report


def test_onnxruntime_benchmark_smoke(tmp_path: Path) -> None:
    """ONNX Runtime benchmark runs on a tiny exported graph."""

    onnx_path = tmp_path / "mobilenetv3_small.onnx"
    output = tmp_path / "benchmark_ort.json"
    assert (
        mobilenetv3.main(
            [
                "export-onnx",
                "--weights",
                "none",
                "--output",
                str(onnx_path),
            ]
        )
        == 0
    )

    exit_code = mobilenetv3.main(
        [
            "benchmark",
            "--runtime",
            "onnxruntime",
            "--onnx-path",
            str(onnx_path),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--image-size",
            "224",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"] == "onnxruntime"
    assert report["latency_ms_p95"] > 0


def test_compare_tsne_smoke(tmp_path: Path) -> None:
    """t-SNE comparison writes CSV and PNG for synthetic images."""

    manifest = _write_manifest(tmp_path, image_count=4)
    checkpoint = tmp_path / "backbone.pt"
    output_dir = tmp_path / "reports"
    backbone = MobileNetV3SmallBackbone(weights="none")
    torch.save({"state_dict": backbone.state_dict()}, checkpoint)

    exit_code = mobilenetv3.main(
        [
            "compare-tsne",
            "--manifest-csv",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--weights",
            "none",
            "--image-size",
            "64",
            "--max-samples",
            "4",
            "--perplexity",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "tsne_embeddings.csv").exists()
    assert (output_dir / "tsne_embeddings.png").exists()


def _write_manifest(tmp_path: Path, image_count: int) -> Path:
    """Create a small image manifest with train and val splits."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    rows: list[dict[str, str]] = []
    for index in range(image_count):
        label = "clear" if index % 2 == 0 else "blocked"
        split = "train" if index < max(2, image_count - 2) else "val"
        image_path = image_dir / f"image_{index}.jpg"
        Image.new("RGB", (80, 80), color=(index * 20 % 255, 30, 120)).save(image_path)
        rows.append({"path": str(image_path), "label": label, "split": split})

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest
