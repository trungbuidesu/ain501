"""Backend TensorBoard cho TrainingLogger."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class _TensorBoardBackend:
    """Backend ghi log vào TensorBoard event files sử dụng torch.utils.tensorboard."""

    def __init__(self, log_dir: str, config: dict[str, Any] | None = None) -> None:
        """Khởi tạo SummaryWriter và ghi config ban đầu nếu có."""
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)
        if config:
            # Ghi hyperparameters dưới dạng text để dễ đọc trong TensorBoard.
            config_text = "\n".join(f"**{k}**: {v}" for k, v in config.items())
            self.writer.add_text("hyperparameters", config_text, global_step=0)
        log.info("TensorBoard logger initialized -> %s", log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Ghi một scalar vào TensorBoard."""
        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Ghi nhiều scalar cùng nhóm vào TensorBoard."""
        self.writer.add_scalars(main_tag, values, global_step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Ghi ảnh TensorBoard, nhận `torch.Tensor` CHW hoặc `numpy` HWC."""
        import torch

        if isinstance(image, torch.Tensor):
            self.writer.add_image(tag, image, global_step=step)
        elif isinstance(image, np.ndarray):
            self.writer.add_image(tag, image, global_step=step, dataformats="HWC")
        else:
            log.warning("TensorBoard: unsupported image type %s", type(image))

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Ghi histogram vào TensorBoard."""
        self.writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Ghi text vào TensorBoard."""
        self.writer.add_text(tag, text, global_step=step)

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Bỏ qua artifact vì TensorBoard không có artifact API native."""
        log.debug(
            "TensorBoard: artifact logging not supported, skipping %s (%s)",
            path,
            artifact_type,
        )

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """Ghi graph kiến trúc model vào TensorBoard."""
        self.writer.add_graph(model, input_sample)

    def finish(self) -> None:
        """Flush và đóng SummaryWriter."""
        self.writer.flush()
        self.writer.close()
        log.info("TensorBoard logger closed.")
