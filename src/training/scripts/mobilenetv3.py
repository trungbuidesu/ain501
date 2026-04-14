# -*- coding: utf-8 -*-
"""CLI for MobileNetV3-Small backbone fine-tuning, export, and benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.training.models.vision import (
    MobileNetV3SmallBackbone,
    MobileNetV3SmallWithHead,
    freeze_mobilenetv3_early_layers,
)
from src.utils import TrainingLogger

DEFAULT_CONFIG = Path("configs/models/mobilenetv3_small.yaml")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated loss and accuracy for one epoch split."""

    loss: float
    accuracy: float


class ImageManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Image classification dataset backed by a `path,label,split` CSV manifest."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        label_to_index: dict[str, int],
        image_size: int,
    ) -> None:
        """Create a manifest dataset with ImageNet-style preprocessing."""

        self.rows = list(rows)
        self.label_to_index = label_to_index
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        """Return number of manifest rows."""

        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load and transform one image row."""

        row = self.rows[index]
        image_path = Path(row["path"])
        label = row["label"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            torch.tensor(self.label_to_index[label], dtype=torch.long),
            str(image_path),
        )


class _FeatureMapExportWrapper(nn.Module):
    """ONNX export wrapper that returns MobileNetV3 feature maps."""

    def __init__(self, backbone: MobileNetV3SmallBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return backbone feature maps."""

        return self.backbone.forward_feature_map(inputs)


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config mapping, returning an empty mapping when missing."""

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _config_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Read a config section as a mapping."""

    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MobileNetV3-Small CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe")
    describe.add_argument("--weights", choices=["default", "imagenet", "none"])
    describe.add_argument("--freeze-until", type=int)

    fine_tune = subparsers.add_parser("fine-tune")
    fine_tune.add_argument("--manifest-csv", type=Path, required=True)
    fine_tune.add_argument("--weights", choices=["default", "imagenet", "none"])
    fine_tune.add_argument("--run-name")
    fine_tune.add_argument("--epochs", type=int)
    fine_tune.add_argument("--batch-size", type=int)
    fine_tune.add_argument("--lr", type=float)
    fine_tune.add_argument("--image-size", type=int)
    fine_tune.add_argument("--freeze-until", type=int)
    fine_tune.add_argument("--device", default="auto")
    fine_tune.add_argument("--checkpoint-dir", type=Path)
    fine_tune.add_argument("--log-dir", type=Path)
    fine_tune.add_argument("--num-workers", type=int, default=0)

    export = subparsers.add_parser("export-onnx")
    export.add_argument("--weights", choices=["default", "imagenet", "none"])
    export.add_argument("--checkpoint", type=Path)
    export.add_argument(
        "--output-kind",
        choices=["embedding", "feature_map"],
        default="embedding",
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--image-size", type=int)
    export.add_argument("--opset", type=int, default=17)

    compare = subparsers.add_parser("compare-tsne")
    compare.add_argument("--manifest-csv", type=Path, required=True)
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--weights", choices=["default", "imagenet", "none"])
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--image-size", type=int)
    compare.add_argument("--max-samples", type=int, default=200)
    compare.add_argument("--perplexity", type=float)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--weights", choices=["default", "imagenet", "none"])
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument(
        "--runtime", choices=["pytorch", "onnxruntime"], default="pytorch"
    )
    benchmark.add_argument("--onnx-path", type=Path)
    benchmark.add_argument(
        "--output-kind", choices=["embedding", "feature_map"], default="embedding"
    )
    benchmark.add_argument("--image-size", type=int)
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--warmup", type=int)
    benchmark.add_argument("--iterations", type=int)
    benchmark.add_argument("--profile-name", default="current-cpu")
    benchmark.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MobileNetV3-Small CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "describe":
        settings = _model_settings(config)
        weights = args.weights or str(settings.get("weights", "default"))
        freeze_until = args.freeze_until or int(settings.get("freeze_until", 7))
        print(json.dumps(describe_model(weights, freeze_until), indent=2))
        return 0

    if args.command == "fine-tune":
        fine_tune_model(args, config)
        return 0

    if args.command == "export-onnx":
        export_onnx(args, config)
        return 0

    if args.command == "compare-tsne":
        compare_tsne(args, config)
        return 0

    if args.command == "benchmark":
        report = benchmark_model(args, config)
        report_text = json.dumps(report, indent=2)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(report_text + "\n", encoding="utf-8")
        print(report_text)
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _model_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the model section from config."""

    return _config_section(config, "model")


def _training_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the training section from config."""

    return _config_section(config, "training")


def _output_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the output section from config."""

    return _config_section(config, "output")


def describe_model(weights: str | None, freeze_until: int) -> dict[str, Any]:
    """Return a JSON-serializable MobileNetV3-Small architecture summary."""

    backbone = MobileNetV3SmallBackbone(weights=weights)
    with_head = MobileNetV3SmallWithHead(backbone, num_classes=2)
    freeze_mobilenetv3_early_layers(with_head, freeze_until=freeze_until)
    total_params = sum(parameter.numel() for parameter in with_head.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in with_head.parameters()
        if parameter.requires_grad
    )
    frozen_params = total_params - trainable_params
    return {
        "architecture": "MobileNetV3-Small",
        "weights": weights or "none",
        "feature_blocks": len(backbone.features),
        "feature_map_shape": [None, backbone.embedding_dim, 7, 7],
        "embedding_shape": [None, backbone.embedding_dim],
        "removed_head": "torchvision classifier MLP",
        "notes": [
            "inverted residual blocks",
            "squeeze-excitation in selected blocks",
            "hard-swish / hard-sigmoid mobile activations",
        ],
        "freeze_until": freeze_until,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def fine_tune_model(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Fine-tune later MobileNetV3 layers with a temporary classifier head."""

    model_settings = _model_settings(config)
    training = _training_settings(config)
    outputs = _output_settings(config)
    weights = args.weights or str(model_settings.get("weights", "default"))
    image_size = args.image_size or int(model_settings.get("image_size", 224))
    freeze_until = args.freeze_until or int(model_settings.get("freeze_until", 7))
    epochs = args.epochs or int(training.get("epochs", 20))
    batch_size = args.batch_size or int(training.get("batch_size", 32))
    learning_rate = args.lr or float(training.get("lr", 1e-4))
    run_name = args.run_name or time.strftime("mobilenetv3_small_%Y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir or Path(
        str(outputs.get("checkpoint_dir", "models/vision/mobilenetv3_small"))
    )
    log_dir = args.log_dir or Path(
        str(outputs.get("log_dir", "runs/mobilenetv3_small"))
    )

    rows = read_manifest_rows(args.manifest_csv)
    label_to_index = label_mapping(rows)
    train_rows = filter_split(rows, "train")
    val_rows = filter_split(rows, "val")
    if not train_rows:
        raise ValueError("manifest must contain at least one train row")

    train_loader = make_loader(
        train_rows, label_to_index, image_size, batch_size, True, args.num_workers
    )
    val_loader = (
        make_loader(
            val_rows, label_to_index, image_size, batch_size, False, args.num_workers
        )
        if val_rows
        else None
    )
    device = resolve_device(args.device)
    backbone = MobileNetV3SmallBackbone(weights=weights)
    model = MobileNetV3SmallWithHead(backbone, num_classes=len(label_to_index)).to(
        device
    )
    freeze_mobilenetv3_early_layers(model, freeze_until=freeze_until)

    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    output_dir = checkpoint_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -math.inf

    with TrainingLogger(
        project="ain501",
        run_name=run_name,
        backends=["tensorboard"],
        log_dir=str(log_dir),
        config={
            "weights": weights,
            "lr": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "freeze_until": freeze_until,
            "image_size": image_size,
            "device": str(device),
        },
    ) as logger:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = (
                run_epoch(model, val_loader, criterion, device, optimizer=None)
                if val_loader is not None
                else EpochMetrics(loss=float("nan"), accuracy=float("nan"))
            )
            logger.log_scalar("train/loss", train_metrics.loss, epoch)
            logger.log_scalar("train/accuracy", train_metrics.accuracy, epoch)
            logger.log_scalar("val/loss", val_metrics.loss, epoch)
            logger.log_scalar("val/accuracy", val_metrics.accuracy, epoch)
            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_metrics.loss:.6f}",
                        f"train_accuracy={train_metrics.accuracy:.6f}",
                        f"val_loss={val_metrics.loss:.6f}",
                        f"val_accuracy={val_metrics.accuracy:.6f}",
                    ]
                )
            )
            score = (
                val_metrics.accuracy
                if val_loader is not None
                else train_metrics.accuracy
            )
            if score > best_accuracy:
                best_accuracy = score
                save_backbone_checkpoint(
                    output_dir / "best_backbone.pt", model, label_to_index, image_size
                )

    save_backbone_checkpoint(
        output_dir / "last_backbone.pt", model, label_to_index, image_size
    )
    (output_dir / "labels.json").write_text(
        json.dumps(label_to_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_checkpoint_dir={output_dir}")


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate a `path,label,split` image manifest."""

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"path", "label", "split"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"manifest must include columns: {sorted(required)}")
    valid_splits = {"train", "val", "test"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} missing columns: {sorted(missing)}")
        if row["split"] not in valid_splits:
            raise ValueError(f"row {index} has unsupported split: {row['split']}")
        if not Path(row["path"]).exists():
            raise ValueError(f"row {index} missing image: {row['path']}")
    return rows


def label_mapping(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    """Build a deterministic label-to-index mapping."""

    labels = sorted({row["label"] for row in rows})
    if not labels:
        raise ValueError("manifest must contain at least one label")
    return {label: index for index, label in enumerate(labels)}


def filter_split(rows: Sequence[dict[str, str]], split: str) -> list[dict[str, str]]:
    """Filter manifest rows by split name."""

    return [row for row in rows if row["split"] == split]


def make_loader(
    rows: Sequence[dict[str, str]],
    label_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]]:
    """Create a DataLoader for manifest rows."""

    dataset = ImageManifestDataset(rows, label_to_index, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return cast(DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]], loader)


def resolve_device(requested: str) -> torch.device:
    """Resolve `cpu`, `cuda`, `xpu`, or `auto` without selecting unsupported XPU."""

    requested = requested.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu"):
        try:
            if (
                torch.xpu.is_available()
                and "arc" in torch.xpu.get_device_name(0).casefold()
            ):
                return torch.device("xpu")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


def run_epoch(
    model: MobileNetV3SmallWithHead,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None,
) -> EpochMetrics:
    """Run one supervised training or evaluation epoch."""

    model.train(optimizer is not None)
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels, _paths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
    if total == 0:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"))
    return EpochMetrics(loss=total_loss / total, accuracy=correct / total)


def save_backbone_checkpoint(
    path: Path,
    model: MobileNetV3SmallWithHead,
    label_to_index: dict[str, int],
    image_size: int,
) -> None:
    """Save only the fine-tuned backbone state and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "mobilenetv3_small_backbone",
            "state_dict": model.backbone.state_dict(),
            "label_to_index": label_to_index,
            "image_size": image_size,
        },
        path,
    )


def load_backbone(
    weights: str | None,
    checkpoint: Path | None = None,
) -> MobileNetV3SmallBackbone:
    """Build a backbone and optionally load a fine-tuned state dict."""

    backbone = MobileNetV3SmallBackbone(weights=weights)
    if checkpoint is None:
        return backbone
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = (
        payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    backbone.load_state_dict(state_dict)
    return backbone


def export_onnx(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Export a FP32 ONNX backbone graph."""

    model_settings = _model_settings(config)
    weights = args.weights or str(model_settings.get("weights", "default"))
    image_size = args.image_size or int(model_settings.get("image_size", 224))
    backbone = load_backbone(weights, args.checkpoint).eval()
    model: nn.Module = (
        backbone
        if args.output_kind == "embedding"
        else _FeatureMapExportWrapper(backbone)
    )
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        args.output,
        input_names=["images"],
        output_names=[args.output_kind],
        dynamic_axes={"images": {0: "batch"}, args.output_kind: {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"wrote_onnx={args.output} output_kind={args.output_kind}")


def compare_tsne(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Compare pretrained and fine-tuned embeddings with t-SNE."""

    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE

    model_settings = _model_settings(config)
    outputs = _output_settings(config)
    weights = args.weights or str(model_settings.get("weights", "default"))
    image_size = args.image_size or int(model_settings.get("image_size", 224))
    output_dir = args.output_dir or Path(
        str(outputs.get("report_dir", "reports/mobilenetv3_small"))
    )
    rows = read_manifest_rows(args.manifest_csv)[: args.max_samples]
    label_to_index = label_mapping(rows)
    loader = make_loader(
        rows, label_to_index, image_size, batch_size=32, shuffle=False, num_workers=0
    )
    pretrained = load_backbone(weights).eval()
    fine_tuned = load_backbone(weights, args.checkpoint).eval()

    pretrained_embeddings, paths, labels = collect_embeddings(pretrained, loader)
    fine_tuned_embeddings, _fine_paths, _fine_labels = collect_embeddings(
        fine_tuned, loader
    )
    embeddings = np.concatenate([pretrained_embeddings, fine_tuned_embeddings], axis=0)
    sample_count = embeddings.shape[0]
    perplexity = args.perplexity or min(30.0, max(2.0, float(sample_count // 3)))
    if perplexity >= sample_count:
        perplexity = max(1.0, float(sample_count - 1))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tsne_embeddings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "path", "label", "x", "y"])
        writer.writeheader()
        for source_index, source in enumerate(("pretrained", "fine_tuned")):
            offset = source_index * len(rows)
            for row_index, path in enumerate(paths):
                writer.writerow(
                    {
                        "source": source,
                        "path": path,
                        "label": labels[row_index],
                        "x": f"{coordinates[offset + row_index, 0]:.8f}",
                        "y": f"{coordinates[offset + row_index, 1]:.8f}",
                    }
                )
    png_path = output_dir / "tsne_embeddings.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(
        coordinates[: len(rows), 0],
        coordinates[: len(rows), 1],
        label="pretrained",
        alpha=0.7,
    )
    plt.scatter(
        coordinates[len(rows) :, 0],
        coordinates[len(rows) :, 1],
        label="fine_tuned",
        alpha=0.7,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"wrote_tsne_csv={csv_path} wrote_tsne_png={png_path}")


def collect_embeddings(
    backbone: MobileNetV3SmallBackbone,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Collect embeddings, image paths, and numeric labels from a loader."""

    embeddings: list[np.ndarray] = []
    paths: list[str] = []
    labels: list[str] = []
    with torch.no_grad():
        for inputs, batch_labels, batch_paths in loader:
            output = backbone.forward_embedding(inputs).cpu().numpy()
            embeddings.append(output)
            paths.extend(str(path) for path in batch_paths)
            labels.extend(str(int(label)) for label in batch_labels)
    if not embeddings:
        raise ValueError("no embeddings collected")
    return np.concatenate(embeddings, axis=0), paths, labels


def benchmark_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark CPU latency and RAM for PyTorch or ONNX Runtime."""

    model_settings = _model_settings(config)
    benchmark = _config_section(config, "benchmark")
    weights = args.weights or str(model_settings.get("weights", "default"))
    image_size = args.image_size or int(model_settings.get("image_size", 224))
    batch_size = args.batch_size or int(benchmark.get("batch_size", 1))
    warmup = args.warmup or int(benchmark.get("warmup", 20))
    iterations = args.iterations or int(benchmark.get("iterations", 100))
    sample = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    start_ram = current_process_rss_mb()

    if args.runtime == "pytorch":
        backbone = load_backbone(weights, args.checkpoint).eval()
        model: nn.Module = (
            backbone
            if args.output_kind == "embedding"
            else _FeatureMapExportWrapper(backbone)
        )
        timings = benchmark_pytorch(model, sample, warmup, iterations)
    else:
        if args.onnx_path is None:
            raise ValueError("--onnx-path is required for onnxruntime benchmark")
        timings = benchmark_onnxruntime(
            args.onnx_path, sample.numpy(), warmup, iterations
        )

    end_ram = current_process_rss_mb()
    return {
        "profile_name": args.profile_name,
        "runtime": args.runtime,
        "output_kind": args.output_kind,
        "cpu": platform.processor() or platform.machine(),
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_p50": statistics.median(timings),
        "latency_ms_p95": percentile(timings, 0.95),
        "ram_mb_start": start_ram,
        "ram_mb_end": end_ram,
        "ram_mb_delta": end_ram - start_ram,
    }


def benchmark_pytorch(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Benchmark PyTorch CPU inference and return per-iteration milliseconds."""

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
    """Benchmark ONNX Runtime CPU inference and return milliseconds."""

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
    """Return a nearest-rank percentile from a non-empty value sequence."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def current_process_rss_mb() -> float:
    """Return current process resident memory in MiB when the platform supports it."""

    if sys.platform == "win32":
        return _windows_rss_mb()
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _windows_rss_mb() -> float:
    """Return Windows process working-set size in MiB via psapi."""

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        """Subset of PROCESS_MEMORY_COUNTERS used for working set size."""

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
    if not ok:
        return float("nan")
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


if __name__ == "__main__":
    sys.exit(main())
