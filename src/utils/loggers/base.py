"""Giao thức (protocol) cho các backend logging dùng chung."""

from __future__ import annotations

from typing import Any, Protocol


class LoggerBackend(Protocol):
    """Giao diện (interface) được triển khai bởi các backend của training logger."""

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...

    def log_scalars(
        self, main_tag: str, values: dict[str, float], step: int
    ) -> None: ...

    def log_image(self, tag: str, image: Any, step: int) -> None: ...

    def log_histogram(self, tag: str, values: Any, step: int) -> None: ...

    def log_text(self, tag: str, text: str, step: int) -> None: ...

    def log_artifact(self, path: str, artifact_type: str = "model") -> None: ...

    def log_model_graph(self, model: Any, input_sample: Any) -> None: ...

    def finish(self) -> None: ...
