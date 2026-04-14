"""Tiện ích validate Piper TTS.

Module kiểm tra config/model voice local và có thể synthesize WAV mẫu để sanity
check âm thanh. Dry-run không render âm thanh thật.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Any

from src.training.data.core import ValidationIssue, ValidationReport, load_yaml


def validate_wav(
    path: Path,
    expected_sample_rate_hz: int | None = None,
) -> ValidationReport:
    """Validate WAV được synthesize từ Piper."""

    issues: list[ValidationIssue] = []
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
    except wave.Error as exc:
        return ValidationReport(1, (ValidationIssue("error", str(path), str(exc)),))

    if frames <= 0:
        issues.append(ValidationIssue("error", str(path), "duration is zero"))
    if channels != 1:
        issues.append(ValidationIssue("error", str(path), "expected mono audio"))
    if sample_width not in {2, 4}:
        issues.append(ValidationIssue("error", str(path), "expected PCM sample width"))
    if expected_sample_rate_hz is not None and sample_rate != expected_sample_rate_hz:
        issues.append(
            ValidationIssue(
                "error",
                str(path),
                f"sample rate {sample_rate} != {expected_sample_rate_hz}",
            )
        )
    return ValidationReport(1, tuple(issues))


def tts_voice_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Trích xuất danh sách voice entries từ TTS config defaults."""

    voices = config.get("voices", {})
    if not isinstance(voices, dict):
        raise ValueError("TTS config missing voices")
    defaults = voices.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("TTS config missing voices.defaults")
    return [entry for entry in defaults.values() if isinstance(entry, dict)]


def validate_tts_config(
    config_path: Path,
    execute: bool = False,
    strict_files: bool = True,
) -> ValidationReport:
    """Validate Piper voice config và tùy chọn synthesize WAV mẫu.

    `strict_files=False` cho phép dry-run trước khi tải model. `execute=True`
    gọi Piper CLI để sinh âm thanh thật.
    """

    config = load_yaml(config_path)
    issues: list[ValidationIssue] = []
    checked_files = 0
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("TTS config missing paths")
    output_root = Path(str(paths.get("validation_output", "data/tts_validation")))

    for voice in tts_voice_entries(config):
        model_path = Path(str(voice["model_path"]))
        voice_config_path = Path(str(voice["config_path"]))
        checked_files += 2
        if not model_path.exists() and strict_files:
            issues.append(
                ValidationIssue("error", str(model_path), "missing model file")
            )
        if not voice_config_path.exists():
            if strict_files:
                issues.append(
                    ValidationIssue(
                        "error",
                        str(voice_config_path),
                        "missing config file",
                    )
                )
            continue
        try:
            json.loads(voice_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue("error", str(voice_config_path), f"invalid json: {exc}")
            )
        if execute and model_path.exists() and voice_config_path.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            output_wav = output_root / f"{voice['id']}.wav"
            synthesize_with_piper(
                model_path=model_path,
                output_wav=output_wav,
                text=str(voice["prompt"]),
            )
            checked_files += 1
            issues.extend(
                validate_wav(
                    output_wav,
                    expected_sample_rate_hz=int(voice["expected_sample_rate_hz"]),
                ).issues
            )
    return ValidationReport(checked_files, tuple(issues))


def synthesize_with_piper(model_path: Path, output_wav: Path, text: str) -> None:
    """Gọi Piper CLI để synthesize một prompt."""

    command = [
        "piper",
        "--model",
        str(model_path),
        "--output_file",
        str(output_wav),
    ]
    subprocess.run(
        command,
        input=text,
        text=True,
        check=True,
        capture_output=True,
    )
