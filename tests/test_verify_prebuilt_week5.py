"""Smoke tests for Week 5 pre-built verification CLI (no heavy model loads)."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.scripts.verify_prebuilt_week5 import REPORT_DIR, build_parser, main


def test_parser_accepts_subcommands() -> None:
    p = build_parser()
    p.parse_args(["paddleocr", "--synthetic", "--limit", "1"])
    p.parse_args(["mediapipe", "--synthetic", "--limit", "1"])
    p.parse_args(["fastvlm", "--skip-model", "--device", "auto"])
    p.parse_args(["tts"])
    p.parse_args(["all", "--synthetic", "--limit", "1", "--skip-model"])


def test_fastvlm_skip_model_writes_benchmark(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["fastvlm", "--skip-model", "--synthetic", "--limit", "1"])
    assert rc == 0
    bench = json.loads((tmp_path / REPORT_DIR / "fastvlm_benchmark.json").read_text())
    assert bench["status"] == "skipped"
