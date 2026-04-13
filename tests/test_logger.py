# -*- coding: utf-8 -*-
"""Kiểm thử TrainingLogger với TensorBoard backend trên training XPU."""

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import TrainingLogger

LOG_DIR = "runs/_test_logger"


def test_logger_tensorboard_xpu() -> None:
    """Chạy training loop nhỏ, ghi TensorBoard và xác minh event file."""
    # Dọn run test cũ nếu còn tồn tại.
    if Path(LOG_DIR).exists():
        shutil.rmtree(LOG_DIR)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Device: {device}")

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 5),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    config = {"lr": 0.01, "batch_size": 64, "device": str(device)}

    with TrainingLogger(
        project="ain501-test",
        run_name="test-run",
        backends=["tensorboard"],
        log_dir=LOG_DIR,
        config=config,
    ) as logger:

        print(f"Active backends: {logger.active_backends}")
        assert "tensorboard" in logger.active_backends

        x = torch.randn(64, 32, device=device)
        y = torch.randint(0, 5, (64,), device=device)

        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Ghi scalar đơn.
            logger.log_scalar("train/loss", loss.item(), step=epoch)

            # Ghi nhiều metric cùng nhóm.
            acc = (output.argmax(1) == y).float().mean().item()
            logger.log_scalars("metrics", {"loss": loss.item(), "acc": acc}, step=epoch)

            if epoch % 3 == 0:
                print(f"  Epoch {epoch} | loss={loss.item():.4f} | acc={acc:.4f}")

        # Ghi histogram của weights.
        for name, param in model.named_parameters():
            logger.log_histogram(f"weights/{name}", param.data.cpu(), step=9)

        # Ghi text summary.
        logger.log_text("summary", "Test training completed successfully", step=9)

    # Xác minh TensorBoard event file đã được tạo.
    tb_dir = Path(LOG_DIR) / "test-run"
    assert tb_dir.exists(), f"TensorBoard dir not found: {tb_dir}"
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0, "No TensorBoard event files found"

    print(f"\n✓ TensorBoard event files: {[f.name for f in event_files]}")
    print("✓ TrainingLogger test PASSED")

    # Dọn output test.
    shutil.rmtree(LOG_DIR)


def test_logger_log_artifact_uses_artifact_type() -> None:
    """Đảm bảo `log_artifact` dispatch tham số `artifact_type` xuống backend."""

    class FakeBackend:
        """Backend giả để bắt tham số artifact."""

        def __init__(self) -> None:
            """Khởi tạo vùng nhớ cho call gần nhất."""
            self.call: tuple[str, str] | None = None

        def log_artifact(self, path: str, artifact_type: str = "model") -> None:
            """Lưu lại path và artifact type được logger truyền xuống."""
            self.call = (path, artifact_type)

    backend = FakeBackend()
    logger = TrainingLogger(backends=[])
    logger._backends.append(backend)

    logger.log_artifact("models/best.pt", artifact_type="checkpoint")

    assert backend.call == ("models/best.pt", "checkpoint")


if __name__ == "__main__":
    test_logger_tensorboard_xpu()
