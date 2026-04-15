"""Smoke tests for Week 6 registry, rubric, and ONNX feed helpers."""

from __future__ import annotations

import numpy as np

from src.training.scripts.benchmark_suite import (
    load_registry,
    repo_root,
    rubric_decision,
)
from src.training.scripts.verify_onnx_registry import collect_onnx_entries
from src.utils.onnx_benchmark import feeds_from_input_specs


def test_registry_parse() -> None:
    root = repo_root()
    reg = root / "models/registry.json"
    data = load_registry(reg)
    assert data.get("schema_version") == 1
    models = data.get("models", [])
    assert isinstance(models, list)
    assert len(models) >= 1


def test_collect_onnx_entries() -> None:
    root = repo_root()
    data = load_registry(root / "models/registry.json")
    entries = collect_onnx_entries(data)
    assert isinstance(entries, list)
    labels = {e[0] for e in entries}
    assert any(x.startswith("minilm_l6:") for x in labels)


def test_feeds_int64() -> None:
    feeds = feeds_from_input_specs(
        {
            "input_ids": {"shape": [1, 8], "dtype": "int64"},
            "attention_mask": {"shape": [1, 8], "dtype": "int64"},
        }
    )
    assert feeds["input_ids"].dtype == np.int64
    assert feeds["attention_mask"].dtype == np.int64


def test_rubric_prefers_fp32_when_int8_slow() -> None:
    model = {"expected_latency_ms": {"edge_cpu": 10.0}}
    rubric = {"max_absolute_metric_drop": 0.05}
    row = {
        "model_id": "x",
        "metric_drop": 0.0,
        "int8_latency_ms": 100.0,
        "fp32_latency_ms": 20.0,
    }
    decision = rubric_decision(row, model, rubric)
    assert decision["chosen_artifact"] == "fp32_onnx_preferred"


def test_rubric_no_onnx_when_both_latencies_missing() -> None:
    model = {"expected_latency_ms": {"edge_cpu": 10.0}}
    rubric = {"max_absolute_metric_drop": 0.05}
    row = {
        "model_id": "x",
        "metric_drop": None,
        "int8_latency_ms": None,
        "fp32_latency_ms": None,
    }
    decision = rubric_decision(row, model, rubric)
    assert decision["chosen_artifact"] == "no_onnx_benchmark"
