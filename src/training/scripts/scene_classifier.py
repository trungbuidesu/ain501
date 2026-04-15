"""CLI for scene router classifier training, QAT, and ONNX deployment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # type: ignore[import-untyped]
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.training.data import (
    SCENE_ROUTER_LABELS,
    augment_balance_plan,
    build_scene_router_manifest,
    collect_face_detected_rows,
    collect_motion_rows,
    collect_scene_change_rows,
    collect_static_rows,
    collect_text_present_rows,
    write_scene_router_manifest,
)
from src.training.models.vision import (
    MobileNetSceneHead,
    TinySceneCNN,
    build_scene_classifier,
    freeze_mobilenet_scene_backbone,
)
from src.utils import TrainingLogger, resolve_device

DEFAULT_CONFIG = Path("configs/models/scene_classifier.yaml")
DEFAULT_SCENE_MANIFEST = Path("data/processed/scene_accessibility/manifest.csv")
DEFAULT_UCF_FRAMES_MANIFEST = Path("data/processed/action_accessibility/manifest_frames.csv")
DEFAULT_TEXTOCR_TRAIN_ANN = Path("data/external/textocr/TextOCR_0.1_train.json")
DEFAULT_TEXTOCR_VAL_ANN = Path("data/external/textocr/TextOCR_0.1_val.json")
DEFAULT_TEXTOCR_IMAGES = Path("data/external/textocr/train_val_images")
DEFAULT_FACE_ROOTS = (Path("data/external/lfw"), Path("data/external/wider_face"))
DEFAULT_OUTPUT_MANIFEST = Path("data/processed/scene_classifier/manifest.csv")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregate metrics for one epoch."""

    loss: float
    accuracy: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]


@dataclass(frozen=True)
class PredictionRecord:
    """Prediction output for one sample."""

    path: str
    true_label: str
    predicted_label: str


class SceneManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset that reads `path,label,source,split` CSV rows."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        label_to_index: dict[str, int],
        image_size: int,
        train: bool,
    ) -> None:
        self.rows = list(rows)
        self.label_to_index = label_to_index
        transform_steps: list[Any] = [transforms.Resize((image_size, image_size))]
        if train:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.02,
                    ),
                ]
            )
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image_path = Path(row["path"])
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label_idx = self.label_to_index[row["label"]]
        return tensor, torch.tensor(label_idx, dtype=torch.long), str(image_path)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config mapping."""

    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def _config_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build parser for scene classifier command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe")
    describe.add_argument("--variant", choices=["tiny_cnn", "mobilenet_head"])
    describe.add_argument("--weights", choices=["default", "imagenet", "none"])

    prepare_manifest = subparsers.add_parser("prepare-manifest")
    prepare_manifest.add_argument("--scene-manifest", type=Path, default=DEFAULT_SCENE_MANIFEST)
    prepare_manifest.add_argument(
        "--ucf-frames-manifest",
        type=Path,
        default=DEFAULT_UCF_FRAMES_MANIFEST,
    )
    prepare_manifest.add_argument(
        "--textocr-train-ann", type=Path, default=DEFAULT_TEXTOCR_TRAIN_ANN
    )
    prepare_manifest.add_argument("--textocr-val-ann", type=Path, default=DEFAULT_TEXTOCR_VAL_ANN)
    prepare_manifest.add_argument(
        "--textocr-images-root", type=Path, default=DEFAULT_TEXTOCR_IMAGES
    )
    prepare_manifest.add_argument(
        "--face-roots",
        type=Path,
        nargs="*",
        default=list(DEFAULT_FACE_ROOTS),
    )
    prepare_manifest.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    prepare_manifest.add_argument("--min-per-class", type=int, default=500)
    prepare_manifest.add_argument("--max-per-class", type=int, default=1000)
    prepare_manifest.add_argument("--seed", type=int, default=501)
    prepare_manifest.add_argument("--train-ratio", type=float, default=0.8)
    prepare_manifest.add_argument("--val-ratio", type=float, default=0.1)
    prepare_manifest.add_argument("--scene-change-threshold", type=float, default=20.0)
    prepare_manifest.add_argument("--execute", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--manifest-csv", type=Path, required=True)
    train.add_argument("--run-name")
    train.add_argument("--variant", choices=["tiny_cnn", "mobilenet_head"])
    train.add_argument("--weights", choices=["default", "imagenet", "none"])
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--image-size", type=int)
    train.add_argument("--label-smoothing", type=float)
    train.add_argument("--weighted-sampling", action="store_true")
    train.add_argument("--no-weighted-sampling", action="store_true")
    train.add_argument("--device", default="auto")
    train.add_argument("--checkpoint-dir", type=Path)
    train.add_argument("--log-dir", type=Path)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--target-accuracy", type=float, default=0.85)
    train.add_argument("--enforce-target-accuracy", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest-csv", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--device", default="cpu")

    qat = subparsers.add_parser("qat")
    qat.add_argument("--manifest-csv", type=Path, required=True)
    qat.add_argument("--checkpoint", type=Path, required=True)
    qat.add_argument("--output-checkpoint", type=Path, required=True)
    qat.add_argument("--epochs", type=int)
    qat.add_argument("--lr", type=float)
    qat.add_argument("--batch-size", type=int)
    qat.add_argument("--num-workers", type=int, default=0)
    qat.add_argument("--backend", default="fbgemm")
    qat.add_argument("--device", default="cpu")

    export = subparsers.add_parser("export-onnx")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-fp32", type=Path)
    export.add_argument("--output-int8", type=Path)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--verify-parity", action="store_true")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--runtime", choices=["pytorch", "onnxruntime"])
    benchmark.add_argument("--onnx-path", type=Path)
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--image-size", type=int)
    benchmark.add_argument("--warmup", type=int)
    benchmark.add_argument("--iterations", type=int)
    benchmark.add_argument("--profile-name")
    benchmark.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for scene classifier CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "describe":
        print(json.dumps(describe_model(args, config), indent=2))
        return 0
    if args.command == "prepare-manifest":
        report = prepare_manifest(args)
        if args.execute:
            write_scene_router_manifest(report["rows_data"], args.output_csv)
        payload = {key: value for key, value in report.items() if key != "rows_data"}
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "train":
        train_report = train_model(args, config)
        print(json.dumps(train_report, indent=2))
        return 0
    if args.command == "evaluate":
        eval_report = evaluate_model(args)
        print(json.dumps(eval_report, indent=2))
        return 0
    if args.command == "qat":
        qat_report = run_qat(args, config)
        print(json.dumps(qat_report, indent=2))
        return 0
    if args.command == "export-onnx":
        export_report = export_onnx(args, config)
        print(json.dumps(export_report, indent=2))
        return 0
    if args.command == "benchmark":
        benchmark_report = benchmark_model(args, config)
        report_text = json.dumps(benchmark_report, indent=2)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(report_text + "\n", encoding="utf-8")
        print(report_text)
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


def describe_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Describe model variant and parameter counts."""

    model_cfg = _config_section(config, "model")
    classes_container = _config_section(config, "classes")
    class_values = classes_container.get("names", list(SCENE_ROUTER_LABELS))
    class_names = [str(name) for name in class_values] if isinstance(class_values, list) else list(SCENE_ROUTER_LABELS)
    variant = args.variant or str(model_cfg.get("variant", "tiny_cnn"))
    weights = args.weights or str(model_cfg.get("weights", "none"))
    model = build_scene_classifier(variant, num_classes=len(class_names), weights=weights)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "variant": variant,
        "weights": weights,
        "classes": class_names,
        "num_classes": len(class_names),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build balanced rows for all scene router classes."""

    max_pool = max(args.max_per_class * 2, args.min_per_class)
    static_rows = collect_static_rows(args.scene_manifest, max_pool)
    motion_rows = collect_motion_rows(args.ucf_frames_manifest, max_pool)
    text_rows = collect_text_present_rows(
        [args.textocr_train_ann, args.textocr_val_ann],
        [args.textocr_images_root],
        max_pool,
    )
    face_rows = collect_face_detected_rows(args.face_roots, max_pool)
    scene_change_rows = collect_scene_change_rows(
        args.ucf_frames_manifest,
        max_pool,
        diff_threshold=args.scene_change_threshold,
    )
    rows = build_scene_router_manifest(
        static_rows=static_rows,
        motion_rows=motion_rows,
        text_rows=text_rows,
        face_rows=face_rows,
        scene_change_rows=scene_change_rows,
        min_samples_per_class=args.min_per_class,
        max_samples_per_class=args.max_per_class,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_counts = Counter(row["split"] for row in rows)
    label_counts = Counter(row["label"] for row in rows)
    augmentation = augment_balance_plan(rows, target_per_class=args.max_per_class)
    return {
        "dry_run": not args.execute,
        "output_csv": str(args.output_csv),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "augmentation_plan": augmentation,
        "rows_data": rows,
    }


def train_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Train classifier and save best/last checkpoints."""

    model_cfg = _config_section(config, "model")
    train_cfg = _config_section(config, "training")
    output_cfg = _config_section(config, "output")
    variant = args.variant or str(model_cfg.get("variant", "tiny_cnn"))
    weights = args.weights or str(model_cfg.get("weights", "none"))
    image_size = args.image_size or int(model_cfg.get("image_size", 160))
    epochs = args.epochs or int(train_cfg.get("epochs", 20))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    lr = args.lr or float(train_cfg.get("lr", 1e-3))
    seed = args.seed or int(train_cfg.get("seed", 501))
    label_smoothing = args.label_smoothing
    if label_smoothing is None:
        label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    weighted_sampling = bool(train_cfg.get("weighted_sampling", True))
    if args.weighted_sampling:
        weighted_sampling = True
    if args.no_weighted_sampling:
        weighted_sampling = False
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(train_cfg.get("num_workers", 0))
    run_name = args.run_name or time.strftime("scene_classifier_%Y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir or Path(
        str(output_cfg.get("checkpoint_dir", "models/vision/scene_classifier"))
    )
    log_dir = args.log_dir or Path(str(output_cfg.get("log_dir", "runs/scene_classifier")))
    set_random_seed(seed)

    rows = read_manifest_rows(args.manifest_csv)
    rows = ensure_required_splits(rows, seed=seed)
    label_to_index = label_mapping(rows)
    train_rows = filter_split(rows, "train")
    val_rows = filter_split(rows, "val")
    test_rows = filter_split(rows, "test")
    if not train_rows:
        raise ValueError("manifest must contain train rows")
    if not val_rows:
        raise ValueError("manifest must contain val rows")
    if not test_rows:
        raise ValueError("manifest must contain test rows")

    device = resolve_device(args.device)
    model = build_scene_classifier(variant, num_classes=len(label_to_index), weights=weights).to(
        device
    )
    if isinstance(model, MobileNetSceneHead):
        freeze_mobilenet_scene_backbone(model, freeze=False)

    train_loader = build_loader(
        train_rows,
        label_to_index,
        image_size=image_size,
        batch_size=batch_size,
        train=True,
        num_workers=num_workers,
        weighted_sampling=weighted_sampling,
    )
    val_loader = build_loader(
        val_rows,
        label_to_index,
        image_size=image_size,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        weighted_sampling=False,
    )
    test_loader = build_loader(
        test_rows,
        label_to_index,
        image_size=image_size,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        weighted_sampling=False,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    output_dir = checkpoint_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -math.inf
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    with TrainingLogger(
        project="ain501",
        run_name=run_name,
        backends=["tensorboard"],
        log_dir=str(log_dir),
        config={
            "variant": variant,
            "weights": weights,
            "image_size": image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weighted_sampling": weighted_sampling,
            "device": str(device),
        },
    ) as logger:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device, label_to_index, optimizer
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, device, label_to_index, optimizer=None
            )
            logger.log_scalar("train/loss", train_metrics.loss, epoch)
            logger.log_scalar("train/accuracy", train_metrics.accuracy, epoch)
            logger.log_scalar("val/loss", val_metrics.loss, epoch)
            logger.log_scalar("val/accuracy", val_metrics.accuracy, epoch)
            for label in label_to_index:
                logger.log_scalar(
                    f"val/precision_{label}",
                    val_metrics.precision_per_class.get(label, 0.0),
                    epoch,
                )
                logger.log_scalar(
                    f"val/recall_{label}",
                    val_metrics.recall_per_class.get(label, 0.0),
                    epoch,
                )
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                save_checkpoint(
                    best_path,
                    model=model,
                    label_to_index=label_to_index,
                    variant=variant,
                    weights=weights,
                    image_size=image_size,
                )
            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_metrics.loss:.6f}",
                        f"train_acc={train_metrics.accuracy:.6f}",
                        f"val_loss={val_metrics.loss:.6f}",
                        f"val_acc={val_metrics.accuracy:.6f}",
                    ]
                )
            )

    save_checkpoint(
        last_path,
        model=model,
        label_to_index=label_to_index,
        variant=variant,
        weights=weights,
        image_size=image_size,
    )
    test_metrics = run_epoch(
        model, test_loader, criterion, device, label_to_index, optimizer=None
    )
    prediction_records = collect_predictions(model, test_loader, device, label_to_index)
    confusion_csv, misclassified_csv = write_evaluation_reports(
        output_dir,
        prediction_records,
        label_to_index=label_to_index,
    )
    target_accuracy = float(args.target_accuracy)
    target_met = test_metrics.accuracy >= target_accuracy
    summary = {
        "run_name": run_name,
        "variant": variant,
        "checkpoint_dir": str(output_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics.accuracy,
        "target_accuracy": target_accuracy,
        "target_accuracy_met": target_met,
        "test_precision_per_class": test_metrics.precision_per_class,
        "test_recall_per_class": test_metrics.recall_per_class,
        "test_f1_per_class": test_metrics.f1_per_class,
        "confusion_matrix_csv": str(confusion_csv),
        "misclassifications_csv": str(misclassified_csv),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report_dir = Path(str(output_cfg.get("report_dir", "reports/scene_classifier")))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.enforce_target_accuracy and not target_met:
        raise RuntimeError(
            f"target accuracy gate failed: test_accuracy={test_metrics.accuracy:.4f}, "
            f"required={target_accuracy:.4f}"
        )
    return summary


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a checkpoint on one split."""

    rows = ensure_required_splits(read_manifest_rows(args.manifest_csv), seed=501)
    label_to_index = label_mapping(rows)
    selected_rows = filter_split(rows, args.split)
    if not selected_rows:
        raise ValueError(f"no rows found for split={args.split}")
    model, metadata = load_checkpoint(args.checkpoint)
    image_size = int(metadata.get("image_size", 160))
    loader = build_loader(
        selected_rows,
        label_to_index,
        image_size=image_size,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        weighted_sampling=False,
    )
    criterion = nn.CrossEntropyLoss()
    device = resolve_device(args.device)
    model = model.to(device)
    metrics = run_epoch(model, loader, criterion, device, label_to_index, optimizer=None)
    return {
        "split": args.split,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "precision_per_class": metrics.precision_per_class,
        "recall_per_class": metrics.recall_per_class,
        "f1_per_class": metrics.f1_per_class,
        "variant": metadata.get("variant", "unknown"),
        "checkpoint": str(args.checkpoint),
    }


def run_qat(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Run QAT fine-tuning for TinySceneCNN."""

    qat_cfg = _config_section(config, "qat")
    epochs = args.epochs or int(qat_cfg.get("epochs", 3))
    lr = args.lr or float(qat_cfg.get("lr", 1e-4))
    batch_size = args.batch_size or int(_config_section(config, "training").get("batch_size", 32))
    preferred_backend = str(args.backend or qat_cfg.get("backend", "fbgemm"))
    supported_engines = set(torch.backends.quantized.supported_engines)
    backend = preferred_backend
    if backend not in supported_engines:
        fallback_order = ("fbgemm", "qnnpack", "onednn", "x86")
        matched = next((name for name in fallback_order if name in supported_engines), None)
        if matched is None:
            raise ValueError(
                "Unsupported quantization backend and no fallback available: "
                f"{preferred_backend}"
            )
        backend = matched
    torch.backends.quantized.engine = backend

    model, metadata = load_checkpoint(args.checkpoint)
    if not isinstance(model, TinySceneCNN):
        raise ValueError("QAT is currently supported for tiny_cnn variant only")
    rows = read_manifest_rows(args.manifest_csv)
    label_to_index = label_mapping(rows)
    train_rows = filter_split(rows, "train")
    if not train_rows:
        raise ValueError("manifest must contain train rows")

    device = resolve_device(args.device)
    if device.type != "cpu":
        raise ValueError("QAT step runs on CPU in this implementation")

    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    prepared = cast(TinySceneCNN, torch.ao.quantization.prepare_qat(model, inplace=False))
    loader = build_loader(
        train_rows,
        label_to_index,
        image_size=int(metadata.get("image_size", 160)),
        batch_size=batch_size,
        train=True,
        num_workers=args.num_workers,
        weighted_sampling=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(prepared.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        metrics = run_epoch(
            prepared,
            loader,
            criterion,
            device,
            label_to_index,
            optimizer=optimizer,
        )
        print(
            f"qat_epoch={epoch} "
            f"train_loss={metrics.loss:.6f} train_acc={metrics.accuracy:.6f}"
        )
    quantized = cast(TinySceneCNN, torch.ao.quantization.convert(prepared.eval(), inplace=False))
    save_checkpoint(
        args.output_checkpoint,
        model=quantized,
        label_to_index=label_to_index,
        variant=str(metadata.get("variant", "tiny_cnn")),
        weights=str(metadata.get("weights", "none")),
        image_size=int(metadata.get("image_size", 160)),
        quantized=True,
    )
    return {
        "output_checkpoint": str(args.output_checkpoint),
        "epochs": epochs,
        "lr": lr,
        "requested_backend": preferred_backend,
        "backend": backend,
    }


def export_onnx(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Export checkpoint to ONNX FP32 and optionally INT8."""

    output_cfg = _config_section(config, "output")
    output_fp32 = args.output_fp32 or Path(
        str(output_cfg.get("onnx_fp32", "models/scene_classifier_fp32.onnx"))
    )
    output_int8 = args.output_int8 or Path(
        str(output_cfg.get("onnx_int8", "models/scene_classifier_int8.onnx"))
    )
    model, metadata = load_checkpoint(args.checkpoint)
    image_size = int(metadata.get("image_size", 160))
    if bool(metadata.get("quantized", False)):
        raise ValueError("Cannot export quantized PyTorch checkpoint directly to ONNX")
    model = model.eval().cpu()
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    output_fp32.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        output_fp32,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    report: dict[str, Any] = {
        "output_fp32": str(output_fp32),
        "output_int8": None,
        "variant": metadata.get("variant", "unknown"),
    }
    if output_int8 is not None:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_int8.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            model_input=str(output_fp32),
            model_output=str(output_int8),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        report["output_int8"] = str(output_int8)
        report["int8_quantization_scope"] = "MatMul,Gemm"
    if args.verify_parity and report["output_int8"]:
        parity = verify_onnx_parity(model, output_fp32, output_int8, image_size=image_size)
        report["parity"] = parity
    return report


def verify_onnx_parity(
    model: nn.Module,
    fp32_path: Path,
    int8_path: Path,
    image_size: int,
) -> dict[str, Any]:
    """Compare top-1 predictions between PyTorch and ONNX graphs."""

    import onnxruntime as ort

    sample = torch.randn(8, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        pytorch_logits = cast(torch.Tensor, model(sample))
        pytorch_top1 = pytorch_logits.argmax(dim=1).cpu().numpy()
    input_payload = sample.cpu().numpy().astype(np.float32)
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = fp32_session.get_inputs()[0].name
    fp32_logits = fp32_session.run(None, {input_name: input_payload})[0]
    int8_logits = int8_session.run(None, {input_name: input_payload})[0]
    fp32_top1 = np.argmax(fp32_logits, axis=1)
    int8_top1 = np.argmax(int8_logits, axis=1)
    return {
        "pytorch_vs_fp32_top1_agreement": float(np.mean(pytorch_top1 == fp32_top1)),
        "fp32_vs_int8_top1_agreement": float(np.mean(fp32_top1 == int8_top1)),
    }


def benchmark_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark PyTorch or ONNXRuntime inference latency on CPU."""

    model_cfg = _config_section(config, "model")
    benchmark_cfg = _config_section(config, "benchmark")
    runtime = args.runtime or str(benchmark_cfg.get("runtime", "onnxruntime"))
    image_size = args.image_size or int(model_cfg.get("image_size", 160))
    batch_size = args.batch_size or int(benchmark_cfg.get("batch_size", 1))
    warmup = args.warmup or int(benchmark_cfg.get("warmup", 20))
    iterations = args.iterations or int(benchmark_cfg.get("iterations", 100))
    profile_name = args.profile_name or str(benchmark_cfg.get("profile_name", "current-cpu"))
    sample = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    start_ram = current_process_rss_mb()

    if runtime == "pytorch":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for pytorch benchmark")
        model, _metadata = load_checkpoint(args.checkpoint)
        timings = benchmark_pytorch(model.eval().cpu(), sample, warmup, iterations)
    else:
        onnx_path = args.onnx_path
        if onnx_path is None:
            onnx_path = Path(str(_config_section(config, "output").get("onnx_int8", "")))
        if not onnx_path or not onnx_path.exists():
            raise ValueError("--onnx-path is required and must exist for onnxruntime benchmark")
        timings = benchmark_onnxruntime(onnx_path, sample.numpy(), warmup, iterations)

    end_ram = current_process_rss_mb()
    latency_p50 = statistics.median(timings)
    latency_p95 = percentile(timings, 0.95)
    return {
        "profile_name": profile_name,
        "runtime": runtime,
        "cpu": platform.processor() or platform.machine(),
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_p50": latency_p50,
        "latency_ms_p95": latency_p95,
        "target_under_5ms": latency_p50 < 5.0,
        "ram_mb_start": start_ram,
        "ram_mb_end": end_ram,
        "ram_mb_delta": end_ram - start_ram,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
    criterion: nn.Module,
    device: torch.device,
    label_to_index: dict[str, int],
    optimizer: optim.Optimizer | None,
) -> EpochMetrics:
    """Run one train/eval epoch with per-class precision and recall."""

    model.train(optimizer is not None)
    total_loss = 0.0
    total = 0
    correct = 0
    index_to_label = {index: label for label, index in label_to_index.items()}
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels, _paths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = cast(torch.Tensor, model(inputs))
            loss = cast(torch.Tensor, criterion(outputs, labels))
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int((predictions == labels).sum().item())
            for truth, prediction in zip(
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                strict=True,
            ):
                true_label = index_to_label[int(truth)]
                pred_label = index_to_label[int(prediction)]
                if true_label == pred_label:
                    tp[true_label] += 1
                else:
                    fp[pred_label] += 1
                    fn[true_label] += 1
    if total == 0:
        return EpochMetrics(float("nan"), float("nan"), {}, {}, {})
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for label in label_to_index:
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        p_value = float(tp[label] / p_denom) if p_denom else 0.0
        r_value = float(tp[label] / r_denom) if r_denom else 0.0
        precision[label] = p_value
        recall[label] = r_value
        f1[label] = (
            float(2.0 * p_value * r_value / (p_value + r_value))
            if (p_value + r_value) > 0.0
            else 0.0
        )
    return EpochMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
        precision_per_class=precision,
        recall_per_class=recall,
        f1_per_class=f1,
    )


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
    device: torch.device,
    label_to_index: dict[str, int],
) -> list[PredictionRecord]:
    """Collect predictions with paths for post-run reports."""

    index_to_label = {index: label for label, index in label_to_index.items()}
    model.eval()
    records: list[PredictionRecord] = []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device)
            outputs = cast(torch.Tensor, model(inputs))
            predictions = outputs.argmax(dim=1).detach().cpu().tolist()
            true_indices = labels.detach().cpu().tolist()
            for path, truth, pred in zip(paths, true_indices, predictions, strict=True):
                records.append(
                    PredictionRecord(
                        path=str(path),
                        true_label=index_to_label[int(truth)],
                        predicted_label=index_to_label[int(pred)],
                    )
                )
    return records


def write_evaluation_reports(
    output_dir: Path,
    records: Sequence[PredictionRecord],
    *,
    label_to_index: dict[str, int],
) -> tuple[Path, Path]:
    """Write confusion matrix and misclassification CSV reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(label_to_index, key=label_to_index.get)
    matrix: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    misclassified: list[PredictionRecord] = []
    for record in records:
        matrix[record.true_label][record.predicted_label] += 1
        if record.true_label != record.predicted_label:
            misclassified.append(record)

    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *labels])
        for true_label in labels:
            writer.writerow([true_label, *[matrix[true_label][pred] for pred in labels]])

    misclassified_path = output_dir / "misclassifications.csv"
    with misclassified_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "true_label", "predicted_label"],
        )
        writer.writeheader()
        for record in misclassified:
            writer.writerow(
                {
                    "path": record.path,
                    "true_label": record.true_label,
                    "predicted_label": record.predicted_label,
                }
            )
    return confusion_path, misclassified_path


def build_loader(
    rows: Sequence[dict[str, str]],
    label_to_index: dict[str, int],
    *,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
    weighted_sampling: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]]:
    """Build DataLoader with optional weighted class sampling."""

    dataset = SceneManifestDataset(rows, label_to_index, image_size, train=train)
    if train and weighted_sampling:
        counts = Counter(row["label"] for row in rows)
        weights = [1.0 / counts[row["label"]] for row in rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
        return cast(DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]], loader)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
    )
    return cast(DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]], loader)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate manifest rows."""

    if not path.exists():
        raise ValueError(f"manifest file not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"path", "label", "source", "split"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} missing columns: {sorted(missing)}")
        image_path = Path(row["path"])
        if not image_path.exists():
            raise ValueError(f"row {index} missing image: {image_path}")
    return rows


def label_mapping(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    """Build deterministic label mapping from manifest rows."""

    labels = sorted({row["label"] for row in rows})
    if not labels:
        raise ValueError("manifest has no labels")
    return {label: index for index, label in enumerate(labels)}


def filter_split(rows: Sequence[dict[str, str]], split: str) -> list[dict[str, str]]:
    """Filter manifest rows by split name."""

    return [row for row in rows if row["split"] == split]


def ensure_required_splits(
    rows: Sequence[dict[str, str]],
    *,
    seed: int,
) -> list[dict[str, str]]:
    """Ensure train/val/test splits exist for tiny manifests."""

    split_counts = Counter(row["split"] for row in rows)
    if split_counts.get("train", 0) > 0 and split_counts.get("val", 0) > 0 and split_counts.get("test", 0) > 0:
        return list(rows)

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(dict(row))
    rebuilt: list[dict[str, str]] = []
    for label in sorted(by_label):
        bucket = by_label[label]
        rng.shuffle(bucket)
        for index, row in enumerate(bucket):
            copied = dict(row)
            if len(bucket) == 1:
                copied["split"] = "train"
            elif len(bucket) == 2:
                copied["split"] = "train" if index == 0 else "test"
            else:
                if index == 0:
                    copied["split"] = "train"
                elif index == 1:
                    copied["split"] = "val"
                elif index == 2:
                    copied["split"] = "test"
                else:
                    copied["split"] = "train"
            rebuilt.append(copied)

    counts = Counter(row["split"] for row in rebuilt)
    if counts.get("val", 0) == 0:
        for row in rebuilt:
            if row["split"] == "train":
                row["split"] = "val"
                break
    counts = Counter(row["split"] for row in rebuilt)
    if counts.get("test", 0) == 0:
        for row in rebuilt:
            if row["split"] == "train":
                row["split"] = "test"
                break
    return rebuilt


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    label_to_index: dict[str, int],
    variant: str,
    weights: str,
    image_size: int,
    quantized: bool = False,
) -> None:
    """Save model checkpoint with metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "weights": weights,
            "image_size": image_size,
            "label_to_index": label_to_index,
            "quantized": quantized,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load checkpoint and rebuild model object."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    variant = str(payload.get("variant", "tiny_cnn"))
    weights = str(payload.get("weights", "none"))
    label_to_index = payload.get("label_to_index", {})
    if not isinstance(label_to_index, dict):
        raise ValueError("checkpoint label_to_index must be a dict")
    model = build_scene_classifier(
        variant=variant,
        num_classes=max(1, len(label_to_index)),
        weights=weights,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    return model, payload


def benchmark_pytorch(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Benchmark PyTorch CPU inference and return latencies in milliseconds."""

    timings: list[float] = []
    with torch.no_grad():
        for _index in range(warmup):
            _ = model(sample)
        for _index in range(iterations):
            start = time.perf_counter()
            _ = model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def benchmark_onnxruntime(
    onnx_path: Path,
    sample: np.ndarray,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Benchmark ONNX Runtime CPU inference and return latencies in milliseconds."""

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    timings: list[float] = []
    for _index in range(warmup):
        _ = session.run(None, {input_name: sample})
    for _index in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return nearest-rank percentile."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def current_process_rss_mb() -> float:
    """Return process RSS in MiB."""

    if sys.platform == "win32":
        return _windows_rss_mb()
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _windows_rss_mb() -> float:
    """Return process RSS on Windows via psapi."""

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if ok:
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    try:
        import psutil

        process = psutil.Process()
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def set_random_seed(seed: int) -> None:
    """Set all random seeds for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    sys.exit(main())
