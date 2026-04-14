"""Unified training logger facade."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from src.utils.loggers.tensorboard import _TensorBoardBackend
from src.utils.loggers.wandb import _WandbBackend

log = logging.getLogger(__name__)


class TrainingLogger:
    """TrainingLogger thống nhất cho TensorBoard và Weights & Biases.

    Tham số:
        project: Tên project, dùng cho W&B.
        run_name: Tên run hoặc experiment.
        backends: Danh sách backend bật: `tensorboard`, `wandb` hoặc cả hai.
        log_dir: Thư mục log TensorBoard và output local.
        config: Hyperparameters cần log lúc khởi tạo.
        tags: Tags tùy chọn cho W&B.

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

    def _dispatch_to_backends(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> None:
        """Gọi một method trên toàn bộ backend đang hoạt động."""
        for backend in self._backends:
            getattr(backend, method_name)(*args, **kwargs)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Ghi một scalar value vào tất cả backend."""
        self._dispatch_to_backends("log_scalar", tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Ghi nhiều scalar values dưới một grouped tag."""
        self._dispatch_to_backends("log_scalars", main_tag, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Ghi một ảnh, nhận `torch.Tensor` CHW hoặc `numpy` HWC."""
        self._dispatch_to_backends("log_image", tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Ghi histogram của values."""
        self._dispatch_to_backends("log_histogram", tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Ghi chuỗi text."""
        self._dispatch_to_backends("log_text", tag, text, step)

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Ghi file hoặc thư mục như artifact, hiện chỉ W&B hỗ trợ."""
        self._dispatch_to_backends("log_artifact", path, artifact_type)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """Ghi graph kiến trúc model nếu backend hỗ trợ."""
        self._dispatch_to_backends("log_model_graph", model, input_sample)

    def finish(self) -> None:
        """Flush và đóng tất cả backend, gọi ở cuối quá trình huấn luyện."""
        self._dispatch_to_backends("finish")
        self._backends.clear()
        self._names.clear()
        log.info("TrainingLogger finished -> all backends closed.")

    def __enter__(self) -> TrainingLogger:
        """Trả về logger khi dùng context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Đóng logger khi thoát context manager."""
        self.finish()
