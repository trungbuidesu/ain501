from __future__ import annotations

import time

from src.output.audio_output import AudioOutputManager


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return 0

    def terminate(self) -> None:
        self.terminated = True


def test_audio_manager_priority_and_dedup(tmp_path) -> None:
    spoken: list[str] = []

    def synth_fn(text: str):
        spoken.append(text)
        return _FakeProc()

    mgr = AudioOutputManager(
        model_path=tmp_path / "voice.onnx",
        output_dir=tmp_path,
        dedup_window_s=10.0,
        synth_fn=synth_fn,
    )
    assert mgr.enqueue("info text", "info", "sig_info")
    assert mgr.enqueue("hazard text", "hazard", "sig_hazard")
    assert mgr.process_next()
    assert spoken[0] == "hazard text"
    assert mgr.process_next()
    assert spoken[1] == "info text"
    # Dedup on same signature within window.
    assert mgr.enqueue("hazard text", "hazard", "sig_hazard") is False


def test_audio_manager_interrupts_lower_priority(tmp_path) -> None:
    current = _FakeProc()

    def synth_fn(_text: str):
        return _FakeProc()

    mgr = AudioOutputManager(
        model_path=tmp_path / "voice.onnx",
        output_dir=tmp_path,
        dedup_window_s=0.1,
        synth_fn=synth_fn,
    )
    mgr._current_proc = current  # white-box check for interrupt behavior
    mgr._current_rank = 2  # info
    assert mgr.enqueue("hazard now", "hazard", "sig_new")
    assert current.terminated is True
    time.sleep(0.02)
