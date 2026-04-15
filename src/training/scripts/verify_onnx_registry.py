"""Load every ONNX path from models/registry.json and run one forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.training.scripts.benchmark_suite import load_registry, repo_root, resolve_path
from src.utils.onnx_benchmark import (
    create_session,
    feeds_from_input_specs,
    run_single_forward,
)


def collect_onnx_entries(registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    models = registry.get("models", [])
    if not isinstance(models, list):
        return out
    for model in models:
        if not isinstance(model, dict):
            continue
        mid = str(model.get("id", "unknown"))
        artifacts = model.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            continue
        for role in ("fp32_onnx", "int8_onnx"):
            meta = artifacts.get(role)
            if isinstance(meta, dict) and meta.get("path"):
                out.append((f"{mid}:{role}", meta))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("models/registry.json"))
    parser.add_argument("--model", help="Only verify entries for this registry model id")
    args = parser.parse_args(argv)

    root = repo_root()
    reg_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = load_registry(reg_path)
    entries = collect_onnx_entries(registry)
    if args.model:
        prefix = f"{args.model}:"
        entries = [e for e in entries if e[0].startswith(prefix)]
    failures = 0
    for label, meta in entries:
        path = resolve_path(root, str(meta.get("path", "")))
        if path is None or not path.is_file():
            print(f"SKIP missing file: {label} -> {meta.get('path')}")
            continue
        specs = meta.get("input_specs")
        if not isinstance(specs, dict):
            print(f"FAIL no input_specs: {label}", file=sys.stderr)
            failures += 1
            continue
        try:
            feeds = feeds_from_input_specs(specs)
            session = create_session(path)
            _ = run_single_forward(session, feeds)
            print(f"OK {label} {path}")
        except Exception as exc:
            print(f"FAIL {label} {path}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
