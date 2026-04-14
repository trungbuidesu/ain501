"""CLI cho việc fine-tuning, evaluate và export MoViNet-A0.

Module này cung cấp các công cụ để huấn luyện nhận diện hành động trên tập dữ liệu
UCF-101 (subset accessibility), thực hiện Quantization Aware Training (QAT)
và xuất mô hình sang định dạng ONNX.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.data.action_recognition.frame_dataset import FrameClipDataset
from src.training.models.video.movinet import MoViNetA0Backbone
from src.utils import TrainingLogger

DEFAULT_CONFIG = Path("configs/models/movinet_a0.yaml")
DEFAULT_ACTION_MANIFEST = Path("data/processed/action_accessibility/manifest.csv")
DEFAULT_FRAMES_MANIFEST = Path(
    "data/processed/action_accessibility/manifest_frames.csv"
)


def load_config(path: Path) -> dict[str, Any]:
    """Đọc cấu hình từ file YAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Xác định thiết bị tính toán (XPU -> CPU)."""
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class ONNXAdaptiveAvgPoolPatch:
    """Context manager để vá lỗi AdaptiveAvgPool3d khi export sang ONNX.

    MoViNet package sử dụng AdaptiveAvgPool3d, vốn gặp lỗi DispatchError trong
    một số phiên bản PyTorch/ONNX opset khi pooling spatial 172x172 về 1x1.
    """

    def __init__(self) -> None:
        self.original_fn = torch.nn.functional.adaptive_avg_pool3d

    def __enter__(self) -> None:
        def patched_fn(input: torch.Tensor, output_size: Any) -> torch.Tensor:
            # Nếu output_size là [T, 1, 1], ta dùng AvgPool3d với kernel bằng size hiện tại
            if isinstance(output_size, (list, tuple)) and output_size[1:] == (1, 1):
                h, w = input.shape[-2:]
                return torch.nn.functional.avg_pool3d(
                    input, kernel_size=(1, h, w), stride=(1, h, w)
                )
            return self.original_fn(input, output_size)

        torch.nn.functional.adaptive_avg_pool3d = patched_fn

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        torch.nn.functional.adaptive_avg_pool3d = self.original_fn


def get_class_names(manifest_path: Path) -> list[str]:
    """Lấy danh sách các class từ manifest CSV."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        labels = {row["label"] for row in reader}
    return sorted(labels)


class MoViNetCalibrationDataReader:
    """DataReader cho ONNX Runtime Quantization.

    Đọc dữ liệu từ FrameClipDataset để phục vụ quá trình Calibration (cân chỉnh).
    """

    def __init__(
        self,
        dataset: FrameClipDataset,
        batch_size: int = 1,
        max_samples: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_samples = min(max_samples, len(dataset))
        self.device = device
        self.current_idx = 0
        self.input_name = "input"

    def get_next(self) -> dict[str, Any] | None:
        """Trả về batch tiếp theo cho calibrator."""
        if self.current_idx >= self.max_samples:
            return None

        batch_clips = []
        for _ in range(self.batch_size):
            if self.current_idx >= self.max_samples:
                break
            clip, _ = self.dataset[self.current_idx]
            # Preprocess uint8 -> float32 [C, T, H, W]
            clip = clip.float() / 255.0

            from src.training.data.core import (
                DEFAULT_NORMALIZE_MEAN,
                DEFAULT_NORMALIZE_STD,
            )

            # Normalize [C, 1, 1, 1]
            mean = torch.tensor(DEFAULT_NORMALIZE_MEAN).view(3, 1, 1, 1)
            std = torch.tensor(DEFAULT_NORMALIZE_STD).view(3, 1, 1, 1)
            clip = (clip - mean) / std

            batch_clips.append(clip)
            self.current_idx += 1

        if not batch_clips:
            return None

        # Stack to [B, C, T, H, W]
        batch_tensor = torch.stack(batch_clips, dim=0)
        return {self.input_name: batch_tensor.numpy()}

    def rewind(self) -> None:
        """Reset chỉ mục đọc."""
        self.current_idx = 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    logger: Any,
    epoch: int,
) -> tuple[float, float]:
    """Huấn luyện mô hình trong một epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (clips, labels) in enumerate(loader):
        # Chuyển labels lên GPU trước
        labels = labels.to(device, non_blocking=True)
        # Tối ưu: Preprocess clips trực tiếp trên GPU
        clips = preprocess_batch(clips, device)

        optimizer.zero_grad()

        # Với MoViNet causal, cần dọn dẹp buffer trước mỗi video/clip mới
        if hasattr(model, "clean_activation_buffers"):
            model.clean_activation_buffers()

        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 10 == 0:
            step = epoch * len(loader) + i
            logger.log_scalar("train/batch_loss", loss.item(), step=step)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def preprocess_batch(clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Chuyển preprocessing lên GPU để giảm tải CPU."""
    # Clips lúc này là uint8 [B, C, T, H, W]
    clips = clips.to(device, non_blocking=True).float() / 255.0

    # Normalization (đã tính sẵn mean/std chuẩn hóa)
    # Normalization constants cho MoViNet
    # Ta dùng hằng số từ data core
    from src.training.data.core import DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD

    mean = torch.tensor(DEFAULT_NORMALIZE_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(DEFAULT_NORMALIZE_STD, device=device).view(1, 3, 1, 1, 1)

    return (clips - mean) / std


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Đánh giá mô hình trên tập validation/test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for clips, labels in loader:
            labels = labels.to(device, non_blocking=True)
            clips = preprocess_batch(clips, device)

            if hasattr(model, "clean_activation_buffers"):
                model.clean_activation_buffers()

            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), correct / total


def main_describe(args: argparse.Namespace) -> int:
    """In thông tin kiến trúc và cấu hình model."""
    config = load_config(args.config)
    class_names = get_class_names(args.manifest)

    print("--- MoViNet-A0 Architecture Information ---")
    print(f"Variant: {config['model']['variant']}")
    print(f"Causal (Streaming): {config['model']['causal']}")
    print(
        f"Resolution: {config['model']['resolution']}x{config['model']['resolution']}"
    )
    print(f"Action Classes ({len(class_names)}):")
    for i, name in enumerate(class_names, 1):
        print(f"  {i:2d}. {name}")

    model = MoViNetA0Backbone(
        num_classes=len(class_names), causal=config["model"]["causal"]
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    return 0


def main_fine_tune(args: argparse.Namespace) -> int:
    """Thực hiện fine-tuning MoViNet-A0."""
    config = load_config(args.config)
    class_names = get_class_names(args.manifest)
    device = get_device()

    print(f"Starting fine-tune on {device}...")

    # FrameClipDataset — chỉ đọc JPEG, không giải mã video → CPU gần như idle
    frames_manifest = args.frames_manifest
    train_ds = FrameClipDataset(
        manifest_path=frames_manifest,
        class_names=class_names,
        n_clip_frames=config["training"]["n_clip_frames"],
        frame_stride=config["training"]["frame_stride"],
        resolution=config["model"]["resolution"],
        is_train=True,
        temporal_jitter=config["training"]["temporal_jitter"],
    )
    train_ds.rows = [r for r in train_ds.rows if r.get("split", "train") == "train"]

    val_ds = FrameClipDataset(
        manifest_path=frames_manifest,
        class_names=class_names,
        n_clip_frames=config["training"]["n_clip_frames"],
        frame_stride=config["training"]["frame_stride"],
        resolution=config["model"]["resolution"],
        is_train=False,
        temporal_jitter=False,
    )
    val_ds.rows = [r for r in val_ds.rows if r.get("split") == "test"]

    # num_workers=4: JPEG rất nhẹ → 4 worker là dư sức
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model
    model = MoViNetA0Backbone(
        num_classes=len(class_names),
        pretrained_path=config["model"]["pretrained_path"],
        causal=config["model"]["causal"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.CrossEntropyLoss()

    if args.dry_run:
        print("Dry-run requested. Running 1 batch and exiting.")
        model.train()
        clips, labels = next(iter(train_loader))
        clips, labels = clips.to(device), labels.to(device)
        model.clean_activation_buffers()
        outputs = model(clips)
        print(f"Output shape: {outputs.shape}")
        return 0

    # Logging
    with TrainingLogger(
        project="ain501", run_name="movinet_a0_finetune", config=config
    ) as logger:
        for epoch in range(config["training"]["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, logger, epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            logger.log_scalar("train/loss", train_loss, step=epoch)
            logger.log_scalar("train/acc", train_acc, step=epoch)
            logger.log_scalar("val/loss", val_loss, step=epoch)
            logger.log_scalar("val/acc", val_acc, step=epoch)

            # do temporal context
            print(f"Epoch {epoch + 1}/{config['training']['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Lưu checkpoint
            checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # Lưu luôn mỗi epoch cho Phase 0
            torch.save(model.state_dict(), checkpoint_path)

    return 0


def main_evaluate(args: argparse.Namespace) -> int:
    """Đánh giá mô hình và in confusion matrix."""
    config = load_config(args.config)
    class_names = get_class_names(args.manifest)
    device = get_device()

    print(f"Evaluating on {device}...")

    dataset = FrameClipDataset(
        manifest_path=args.frames_manifest,
        class_names=class_names,
        n_clip_frames=config["training"]["n_clip_frames"],
        frame_stride=config["training"]["frame_stride"],
        resolution=config["model"]["resolution"],
        is_train=False,
        temporal_jitter=False,
    )
    dataset.rows = [r for r in dataset.rows if r.get("split") == "test"]
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = MoViNetA0Backbone(
        num_classes=len(class_names), causal=config["model"]["causal"]
    ).to(device)
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, evaluating with random head.")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in loader:
            clips = preprocess_batch(clips, device)
            if hasattr(model, "clean_activation_buffers"):
                model.clean_activation_buffers()
            outputs = model(clips)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    acc = float((all_preds_np == all_labels_np).mean())
    print(f"Top-1 Accuracy: {acc:.4f}")

    # Simple confusion matrix summary
    if args.error_analysis:
        return main_error_analysis(all_labels_np, all_preds_np, class_names)

    return 0


def main_error_analysis(
    labels: np.ndarray, preds: np.ndarray, class_names: list[str]
) -> int:
    """Phân tích các cặp class hay bị nhầm lẫn nhất."""
    from collections import Counter

    errors = [
        (labels[i], preds[i]) for i in range(len(labels)) if labels[i] != preds[i]
    ]
    error_counts = Counter(errors)

    print("\n--- Error Analysis (Top 10 confused pairs) ---")
    for (label_idx, pred_idx), count in error_counts.most_common(10):
        print(f"  {class_names[label_idx]:15s} -> {class_names[pred_idx]:15s}: {count}")

    return 0


def main_export_onnx(args: argparse.Namespace) -> int:
    """Xuất mô hình sang định dạng ONNX."""
    config = load_config(args.config)
    class_names = get_class_names(args.manifest)
    device = torch.device("cpu")  # Export thường trên CPU

    model = MoViNetA0Backbone(
        num_classes=len(class_names), causal=config["model"]["causal"]
    ).to(device)

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best.pt"
    if checkpoint_path.exists():
        sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Parameter count in SD: {len(sd)}")
    else:
        print("Warning: No checkpoint found. Exporting with random weights.")

    model.eval()
    # Kiểm tra xem params có thực sự tồn tại và có giá trị không
    first_param = next(model.parameters())
    print(f"First param mean: {first_param.mean().item():.6f}")

    # Dummy input [B, C, T, H, W]
    dummy_input = torch.randn(
        1,
        3,
        config["training"]["n_clip_frames"],
        config["model"]["resolution"],
        config["model"]["resolution"],
    ).to(device)

    output_path = Path(config["output"]["onnx_fp32"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX to {output_path}...")
    with ONNXAdaptiveAvgPoolPatch():
        # Sử dụng phương pháp truyền thống nhất có thể
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,  # Nâng cấp opset để hỗ trợ tốt hơn
        )

    # Kiểm tra kích thước sau export
    actual_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Model exported successfully. Size: {actual_size:.2f} MB")
    if actual_size < 5:
        print("Warning: Exported model is suspiciously small (~1.7MB). ")
        print("This usually means weights were not included.")
    return 0


def main_ptq(args: argparse.Namespace) -> int:
    """Thực hiện Post-Training Quantization (PTQ) bằng ONNX Runtime."""
    config = load_config(args.config)
    class_names = get_class_names(args.manifest)

    fp32_onnx_path = Path(config["output"]["onnx_fp32"])
    int8_onnx_path = Path(config["output"]["onnx_int8"])

    if not fp32_onnx_path.exists():
        print(f"FP32 ONNX not found at {fp32_onnx_path}. Exporting first...")
        main_export_onnx(args)

    print(f"Starting PTQ: {fp32_onnx_path} -> {int8_onnx_path}")

    # Prepare calibration data
    ds = FrameClipDataset(
        manifest_path=args.frames_manifest,
        class_names=class_names,
        n_clip_frames=config["training"]["n_clip_frames"],
        frame_stride=config["training"]["frame_stride"],
        resolution=config["model"]["resolution"],
        is_train=False,
        temporal_jitter=False,
    )
    # Lấy subset validation để calibrate
    ds.rows = [r for r in ds.rows if r.get("split") == "test"][:100]

    dr = MoViNetCalibrationDataReader(ds, batch_size=1, max_samples=100)

    from onnxruntime.quantization import (
        QuantFormat,
        QuantType,
        quantize_static,
    )

    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,  # QDQ thường tốt hơn cho Intel/OpenVINO
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    print(f"PTQ complete. INT8 model saved at {int8_onnx_path}")

    # So sánh dung lượng
    fp32_size = fp32_onnx_path.stat().st_size / (1024 * 1024)
    int8_size = int8_onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n--- Model Stats ---")
    print(f"  FP32 ONNX Size: {fp32_size:.2f} MB")
    print(f"  INT8 ONNX Size: {int8_size:.2f} MB")
    print(f"  Size Ratio: {int8_size / fp32_size:.2x} (Note: FP32 export in PT2.5 is highly optimized)")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng trình phân tích đối số CLI."""
    parser = argparse.ArgumentParser(description="MoViNet-A0 CLI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ACTION_MANIFEST)
    parser.add_argument(
        "--frames-manifest",
        type=Path,
        default=DEFAULT_FRAMES_MANIFEST,
        help="Manifest frames JPEG (sau khi chạy preprocess_frames.py)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe")

    fine_tune = subparsers.add_parser("fine-tune")
    fine_tune.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--error-analysis", action="store_true")

    subparsers.add_parser("error-analysis")
    subparsers.add_parser("ptq")
    subparsers.add_parser("export-onnx")

    return parser


def main() -> int:
    """Entry point chính."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "describe":
        return main_describe(args)
    if args.command == "fine-tune":
        return main_fine_tune(args)
    if args.command == "evaluate":
        return main_evaluate(args)
    if args.command == "error-analysis":
        # Giả lập bằng cách gọi evaluate với flag
        args.error_analysis = True
        return main_evaluate(args)
    if args.command == "ptq":
        return main_ptq(args)
    if args.command == "export-onnx":
        return main_export_onnx(args)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
