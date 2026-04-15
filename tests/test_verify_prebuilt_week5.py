"""Smoke tests for Week 5 pre-built verification CLI (no heavy model loads)."""

from __future__ import annotations

from src.training.scripts.verify_prebuilt_week5 import build_parser


def test_parser_accepts_subcommands() -> None:
    p = build_parser()
    p.parse_args(["paddleocr", "--synthetic", "--limit", "1"])
    p.parse_args(["mediapipe", "--synthetic", "--limit", "1"])
    p.parse_args(["tts"])
    p.parse_args(["all", "--synthetic", "--limit", "1"])
