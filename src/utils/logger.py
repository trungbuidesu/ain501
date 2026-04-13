# -*- coding: utf-8 -*-
"""Logger huấn luyện thống nhất cho TensorBoard và Weights & Biases.

Cách dùng:
    from src.utils import TrainingLogger

    logger = TrainingLogger(
        project="ain501-yolo",
        run_name="exp-001",
        backends=["tensorboard", "wandb"],
        log_dir="runs/exp-001",
        config={"lr": 1e-3, "epochs": 100},
    )

    for epoch in range(100):
        loss = train_one_epoch(...)
        logger.log_scalar("train/loss", loss, step=epoch)
        logger.log_scalars("metrics", {"acc": 0.95, "f1": 0.92}, step=epoch)

    logger.log_image("predictions", image_tensor, step=epoch)
    logger.log_artifact("model.pt", artifact_type="model")
    logger.finish()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

log = logging.getLogger(__name__)


class _TensorBoardBackend:
    """Backend ghi log TensorBoard."""

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


class _WandbBackend:
    """Backend ghi log Weights & Biases."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        log_dir: str | None = None,
    ) -> None:
        """Khởi tạo W&B run với project, config và tag đã cho."""
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            dir=log_dir,
            reinit=True,
        )
        log.info("W&B logger initialized -> project=%s, run=%s", project, run_name)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Ghi một scalar vào W&B."""
        self._wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Ghi nhiều scalar cùng nhóm vào W&B."""
        prefixed = {f"{main_tag}/{k}": v for k, v in values.items()}
        self._wandb.log(prefixed, step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Ghi ảnh vào W&B."""
        import torch

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        img = self._wandb.Image(image)
        self._wandb.log({tag: img}, step=step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Ghi histogram vào W&B."""
        import torch

        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        hist = self._wandb.Histogram(values)
        self._wandb.log({tag: hist}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Ghi text vào W&B."""
        self._wandb.log({tag: text}, step=step)

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Ghi file hoặc thư mục artifact vào W&B."""
        artifact = self._wandb.Artifact(
            name=Path(path).stem,
            type=artifact_type,
        )
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self._wandb.log_artifact(artifact)
        log.info("W&B: logged artifact %s (type=%s)", path, artifact_type)

    def log_model_graph(self, model: Any, _input_sample: Any) -> None:
        """Theo dõi model bằng W&B; sample input không dùng ở backend này."""
        self._wandb.watch(model, log="all")

    def finish(self) -> None:
        """Kết thúc W&B run hiện tại."""
        self._wandb.finish()
        log.info("W&B logger closed.")


class TrainingLogger:
    """Logger huấn luyện thống nhất, dispatch sang TensorBoard và/hoặc W&B.

    Tham số:
        project: Tên project, dùng cho W&B.
        run_name: Tên run hoặc experiment.
        backends: Danh sách backend bật, gồm "tensorboard" hoặc "wandb".
        log_dir: Thư mục log TensorBoard và output local.
        config: Hyperparameters cần log lúc khởi tạo.
        tags: Tag tùy chọn cho W&B.

    Ví dụ:
        >>> logger = TrainingLogger(
        ...     project="ain501",
        ...     run_name="yolo-v1",
        ...     backends=["tensorboard"],
        ...     log_dir="runs/yolo-v1",
        ...     config={"lr": 0.001, "batch_size": 16},
        ... )
        >>> logger.log_scalar("train/loss", 0.5, step=1)
        >>> logger.finish()
    """

    VALID_BACKENDS: ClassVar[set[str]] = {"tensorboard", "wandb"}

    def __init__(
        self,
        project: str = "ain501",
        run_name: str | None = None,
        backends: list[str] | None = None,
        log_dir: str = "runs",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Khởi tạo logger và các backend được yêu cầu."""
        if backends is None:
            backends = ["tensorboard"]

        invalid = set(backends) - self.VALID_BACKENDS
        if invalid:
            raise ValueError(f"Invalid backends: {invalid}. Use: {self.VALID_BACKENDS}")

        self._backends: list[Any] = []
        self._names: list[str] = []
        full_log_dir = str(Path(log_dir) / run_name) if run_name else log_dir

        for backend_name in backends:
            try:
                if backend_name == "tensorboard":
                    self._backends.append(
                        _TensorBoardBackend(log_dir=full_log_dir, config=config)
                    )
                    self._names.append("tensorboard")

                elif backend_name == "wandb":
                    self._backends.append(
                        _WandbBackend(
                            project=project,
                            run_name=run_name,
                            config=config,
                            tags=tags,
                            log_dir=full_log_dir,
                        )
                    )
                    self._names.append("wandb")

            except ImportError as e:
                log.warning(
                    "Backend '%s' not available (missing dependency): %s",
                    backend_name,
                    e,
                )
            except Exception as e:
                log.warning("Failed to init backend '%s': %s", backend_name, e)

        log.info("TrainingLogger ready -> active backends: %s", self._names)

    @property
    def active_backends(self) -> list[str]:
        """Trả về danh sách backend đang hoạt động."""
        return list(self._names)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Ghi một scalar value vào tất cả backend."""
        for b in self._backends:
            b.log_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Ghi nhiều scalar value dưới một grouped tag."""
        for b in self._backends:
            b.log_scalars(main_tag, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Ghi một ảnh, nhận `torch.Tensor` CHW hoặc `numpy` HWC."""
        for b in self._backends:
            b.log_image(tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Ghi histogram của values."""
        for b in self._backends:
            b.log_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Ghi chuỗi text."""
        for b in self._backends:
            b.log_text(tag, text, step)

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Ghi file hoặc thư mục như artifact, hiện chỉ W&B hỗ trợ."""
        for b in self._backends:
            b.log_artifact(path, artifact_type)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """Ghi graph kiến trúc model nếu backend hỗ trợ."""
        for b in self._backends:
            b.log_model_graph(model, input_sample)

    def finish(self) -> None:
        """Flush và đóng tất cả backend, gọi ở cuối quá trình huấn luyện."""
        for b in self._backends:
            b.finish()
        self._backends.clear()
        self._names.clear()
        log.info("TrainingLogger finished -> all backends closed.")

    def __enter__(self) -> TrainingLogger:
        """Trả về logger khi dùng context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Đóng logger khi thoát context manager."""
        self.finish()
