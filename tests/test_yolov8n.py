# -*- coding: utf-8 -*-
"""Tests for YOLOv8n code-first CLI utilities."""

from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from _pytest.capture import CaptureFixture
from PIL import Image

from src.training.scripts import yolov8n


def test_describe_contains_architecture_fields(capsys: CaptureFixture[str]) -> None:
    """Describe prints the checklist architecture summary."""

    exit_code = yolov8n.main(["describe"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "yolov8n"
    assert report["detection"]["anchor_free"] is True
    assert report["detection"]["loss_components"] == ["box", "cls", "dfl"]
    assert report["detection"]["postprocess"].endswith("NMS")


def test_write_data_yaml_dry_run_and_execute(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """data.yaml generation is dry-run by default and writes only with execute."""

    dataset_root = _write_tiny_yolo_dataset(tmp_path)
    output = tmp_path / "data.yaml"

    exit_code = yolov8n.main(
        [
            "write-data-yaml",
            "--dataset-root",
            str(dataset_root),
            "--classes",
            str(dataset_root / "classes.txt"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert not output.exists()
    dry_run_report = json.loads(capsys.readouterr().out)
    assert dry_run_report["dry_run"] is True
    assert dry_run_report["data"]["nc"] == 2

    exit_code = yolov8n.main(
        [
            "write-data-yaml",
            "--dataset-root",
            str(dataset_root),
            "--classes",
            str(dataset_root / "classes.txt"),
            "--output",
            str(output),
            "--execute",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["path"] == str(dataset_root)
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["nc"] == 2
    assert payload["names"] == ["person", "dog"]


def test_train_evaluate_predict_and_ablate_are_dry_run_safe(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Dry-run command surface should not require Ultralytics execution."""

    dataset_root = _write_tiny_yolo_dataset(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    config = _write_config(tmp_path, dataset_root, data_yaml)
    assert yolov8n.main(["--config", str(config), "write-data-yaml", "--execute"]) == 0
    capsys.readouterr()

    assert yolov8n.main(["--config", str(config), "train", "--epochs", "1"]) == 0
    train_report = json.loads(capsys.readouterr().out)
    assert train_report["dry_run"] is True
    assert train_report["train"]["epochs"] == 1

    assert yolov8n.main(["--config", str(config), "evaluate"]) == 0
    evaluate_report = json.loads(capsys.readouterr().out)
    assert evaluate_report["dry_run"] is True
    assert evaluate_report["evaluate"]["data"] == str(data_yaml)

    errors_csv = tmp_path / "errors.csv"
    assert (
        yolov8n.main(
            [
                "--config",
                str(config),
                "predict-errors",
                "--output-csv",
                str(errors_csv),
            ]
        )
        == 0
    )
    predict_report = json.loads(capsys.readouterr().out)
    assert predict_report["dry_run"] is True
    assert not errors_csv.exists()

    matrix_output = tmp_path / "ablation.json"
    assert (
        yolov8n.main(
            [
                "--config",
                str(config),
                "ablate",
                "--matrix-output",
                str(matrix_output),
            ]
        )
        == 0
    )
    ablation_report = json.loads(capsys.readouterr().out)
    assert ablation_report["dry_run"] is True
    assert len(ablation_report["ablation"]) == 4
    assert matrix_output.exists()


def test_qat_guard_reports_unsupported(capsys: CaptureFixture[str]) -> None:
    """QAT command reports the explicit Phase 0 guard."""

    exit_code = yolov8n.main(["qat"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["supported"] is False
    assert "Ultralytics YOLOv8n" in report["method"]


def _write_tiny_yolo_dataset(tmp_path: Path) -> Path:
    """Create a tiny canonical YOLO dataset fixture."""

    root = tmp_path / "yolo"
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_path = image_dir / f"{split}_0.jpg"
        Image.new("RGB", (64, 64), color=(120, 30, 40)).save(image_path)
        (label_dir / f"{split}_0.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.250000\n",
            encoding="utf-8",
        )
    (root / "classes.txt").write_text("person\ndog\n", encoding="utf-8")
    return root


def _write_config(tmp_path: Path, dataset_root: Path, data_yaml: Path) -> Path:
    """Write a YOLOv8n test config that points at temp fixtures."""

    config = tmp_path / "yolov8n.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"weights": "yolov8n.pt"},
                "dataset": {
                    "root": str(dataset_root),
                    "classes": str(dataset_root / "classes.txt"),
                    "data_yaml": str(data_yaml),
                },
                "training": {
                    "imgsz": 64,
                    "epochs": 2,
                    "batch": 1,
                    "lr0": 0.01,
                    "mosaic": 1.0,
                    "mixup": 0.1,
                    "multi_scale": True,
                },
                "output": {
                    "run_dir": str(tmp_path / "runs"),
                    "report_dir": str(tmp_path / "reports"),
                    "checkpoint_dir": str(tmp_path / "models"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config
