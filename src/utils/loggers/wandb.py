"""Backend Weights & Biases cho TrainingLogger."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class _WandbBackend:
    """Backend ghi log vào Weights & Biases cloud sử dụng wandb SDK."""

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
