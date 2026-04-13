"""
Unified Training Logger — shared utility for TensorBoard & Weights & Biases.

Usage:
    from src.utils import TrainingLogger

    logger = TrainingLogger(
        project="ain501-yolo",
        run_name="exp-001",
        backends=["tensorboard", "wandb"],  # or just one
        log_dir="runs/exp-001",
        config={"lr": 1e-3, "epochs": 100},
    )

    for epoch in range(100):
        loss = train_one_epoch(...)
        logger.log_scalar("train/loss", loss, step=epoch)
        logger.log_scalars("metrics", {"acc": 0.95, "f1": 0.92}, step=epoch)

    logger.log_image("predictions", image_tensor, step=epoch)
    logger.log_artifact("model.pt", type="model")
    logger.finish()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class _TensorBoardBackend:
    """TensorBoard logging backend."""

    def __init__(self, log_dir: str, config: dict[str, Any] | None = None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)
        if config:
            # Log hyperparameters as text
            config_text = "\n".join(f"**{k}**: {v}" for k, v in config.items())
            self.writer.add_text("hyperparameters", config_text, global_step=0)
        log.info("TensorBoard logger initialized → %s", log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        self.writer.add_scalars(main_tag, values, global_step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log image. Accepts torch.Tensor (C,H,W) or numpy (H,W,C)."""
        import torch

        if isinstance(image, torch.Tensor):
            self.writer.add_image(tag, image, global_step=step)
        elif isinstance(image, np.ndarray):
            self.writer.add_image(tag, image, global_step=step, dataformats="HWC")
        else:
            log.warning("TensorBoard: unsupported image type %s", type(image))

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self.writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, global_step=step)

    def log_artifact(self, path: str, type: str = "file") -> None:
        # TensorBoard doesn't have native artifact support
        log.debug("TensorBoard: artifact logging not supported, skipping %s", path)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        self.writer.add_graph(model, input_sample)

    def finish(self) -> None:
        self.writer.flush()
        self.writer.close()
        log.info("TensorBoard logger closed.")


class _WandbBackend:
    """Weights & Biases logging backend."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        log_dir: str | None = None,
    ) -> None:
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
        log.info("W&B logger initialized → project=%s, run=%s", project, run_name)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        prefixed = {f"{main_tag}/{k}": v for k, v in values.items()}
        self._wandb.log(prefixed, step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        import torch

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        img = self._wandb.Image(image)
        self._wandb.log({tag: img}, step=step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        import torch

        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        hist = self._wandb.Histogram(values)
        self._wandb.log({tag: hist}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._wandb.log({tag: text}, step=step)

    def log_artifact(self, path: str, type: str = "model") -> None:
        artifact = self._wandb.Artifact(
            name=Path(path).stem,
            type=type,
        )
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self._wandb.log_artifact(artifact)
        log.info("W&B: logged artifact %s (type=%s)", path, type)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        self._wandb.watch(model, log="all")

    def finish(self) -> None:
        self._wandb.finish()
        log.info("W&B logger closed.")


class TrainingLogger:
    """
    Unified training logger that dispatches to TensorBoard and/or W&B.

    Args:
        project: Project name (used by W&B).
        run_name: Run/experiment name.
        backends: List of backends to enable. Options: "tensorboard", "wandb".
        log_dir: Directory for TensorBoard logs and local output.
        config: Hyperparameters dict to log at init.
        tags: Optional tags (W&B only).

    Example:
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

    VALID_BACKENDS = {"tensorboard", "wandb"}

    def __init__(
        self,
        project: str = "ain501",
        run_name: str | None = None,
        backends: list[str] | None = None,
        log_dir: str = "runs",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if backends is None:
            backends = ["tensorboard"]

        invalid = set(backends) - self.VALID_BACKENDS
        if invalid:
            raise ValueError(f"Invalid backends: {invalid}. Use: {self.VALID_BACKENDS}")

        self._backends: list[Any] = []
        self._names: list[str] = []

        # Resolve log directory
        if run_name:
            full_log_dir = str(Path(log_dir) / run_name)
        else:
            full_log_dir = log_dir

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

        log.info("TrainingLogger ready — active backends: %s", self._names)

    @property
    def active_backends(self) -> list[str]:
        """Return list of currently active backend names."""
        return list(self._names)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        for b in self._backends:
            b.log_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars under a grouped tag."""
        for b in self._backends:
            b.log_scalars(main_tag, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log an image (torch.Tensor CHW or numpy HWC)."""
        for b in self._backends:
            b.log_image(tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram of values."""
        for b in self._backends:
            b.log_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string."""
        for b in self._backends:
            b.log_text(tag, text, step)

    def log_artifact(self, path: str, type: str = "model") -> None:
        """Log a file/directory as an artifact (W&B only)."""
        for b in self._backends:
            b.log_artifact(path, type)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """Log model architecture graph."""
        for b in self._backends:
            b.log_model_graph(model, input_sample)

    def finish(self) -> None:
        """Flush and close all backends. Call at end of training."""
        for b in self._backends:
            b.finish()
        self._backends.clear()
        self._names.clear()
        log.info("TrainingLogger finished — all backends closed.")

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()
