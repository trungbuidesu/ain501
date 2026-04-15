"""Validate deployable model files before ONNX Runtime sessions."""

from __future__ import annotations

from pathlib import Path


def require_file(
    path: Path | str,
    *,
    model_id: str,
    hint: str = "",
) -> Path:
    """Fail fast with a clear message if an expected artifact is missing."""

    p = Path(path)
    if p.is_file():
        return p.resolve()
    extra = f" {hint}" if hint else ""
    raise FileNotFoundError(
        f"Missing artifact for model '{model_id}': expected file at {p.resolve()}. "
        f"Check configs and models/registry.json; place the exported ONNX or download "
        f"the file into the repo without modifying models/ training scripts.{extra}"
    )
