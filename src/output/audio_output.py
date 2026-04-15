"""Priority speech queue with interrupt and dedup for Piper TTS."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any

PRIORITY_RANK = {"hazard": 0, "navigation": 1, "info": 2}
DEBUG_LOG_PATH = Path("debug-fe6f6f.log")
DEBUG_SESSION_ID = "fe6f6f"


def _debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


@dataclass(order=True, slots=True)
class SpeechMessage:
    rank: int
    created_at: float
    text: str = field(compare=False)
    priority: str = field(compare=False)
    signature: str = field(compare=False)


class AudioOutputManager:
    """Audio output manager for Piper with priority, preemption, and dedup."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        output_dir: Path | str = Path("reports/phase2/audio"),
        dedup_window_s: float = 2.5,
        synth_fn: Callable[[str], Any] | None = None,
        debug_run_id: str | None = None,
        piper_executable: str = "piper",
    ) -> None:
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_window_s = float(dedup_window_s)
        self._queue: list[SpeechMessage] = []
        self._lock = threading.Lock()
        self._last_spoken_ts: dict[str, float] = {}
        self._stop = False
        self._worker: threading.Thread | None = None
        self._current_proc: Any | None = None
        self._current_rank: int | None = None
        self._synth_fn = synth_fn if synth_fn is not None else self._spawn_piper
        self._uses_builtin_piper = synth_fn is None
        self._debug_run_id = debug_run_id or f"audio_{int(time.time() * 1000)}"
        self._piper_executable = piper_executable
        self._enabled = True
        self._disabled_reason: str | None = None

    def start(self) -> None:
        if self._worker is not None:
            return
        if self._uses_builtin_piper:
            resolved = which(self._piper_executable)
            model_exists = self.model_path.exists()
            if resolved is None or not model_exists:
                self._enabled = False
                reasons: list[str] = []
                if resolved is None:
                    reasons.append("piper_executable_not_found")
                if not model_exists:
                    reasons.append("piper_model_missing")
                self._disabled_reason = ",".join(reasons)
                # region agent log
                _debug_log(
                    run_id=self._debug_run_id,
                    hypothesis_id="H6_audio_preflight_disable",
                    location="src/output/audio_output.py:96",
                    message="Audio disabled by preflight checks",
                    data={
                        "reasons": reasons,
                        "piper_executable": self._piper_executable,
                        "piper_executable_resolved": resolved,
                        "model_path": str(self.model_path),
                        "model_exists": model_exists,
                    },
                )
                # endregion
                return
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop = True
        with self._lock:
            proc = self._current_proc
            self._current_proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def enqueue(self, text: str, priority: str, signature: str) -> bool:
        if not self._enabled:
            return False
        rank = PRIORITY_RANK.get(priority, PRIORITY_RANK["info"])
        now = time.monotonic()
        last = self._last_spoken_ts.get(signature)
        if last is not None and (now - last) < self.dedup_window_s:
            return False

        msg = SpeechMessage(
            rank=rank,
            created_at=now,
            text=text,
            priority=priority,
            signature=signature,
        )
        with self._lock:
            # Preempt lower-priority speech.
            if self._current_proc is not None and self._current_rank is not None:
                if rank < self._current_rank:
                    try:
                        self._current_proc.terminate()
                    except Exception:
                        pass
            self._queue.append(msg)
            self._queue.sort()
        return True

    def process_next(self) -> bool:
        with self._lock:
            if not self._queue:
                return False
            msg = self._queue.pop(0)
            self._current_rank = msg.rank
            queue_size = len(self._queue)
        # region agent log
        _debug_log(
            run_id=self._debug_run_id,
            hypothesis_id="H3_queue_worker_crash",
            location="src/output/audio_output.py:135",
            message="Dequeued speech message for synthesis",
            data={
                "priority": msg.priority,
                "signature": msg.signature,
                "queue_size_after_pop": queue_size,
            },
        )
        # endregion
        try:
            proc = self._synth_fn(msg.text)
        except Exception as exc:
            # region agent log
            _debug_log(
                run_id=self._debug_run_id,
                hypothesis_id="H7_synth_exception_guarded",
                location="src/output/audio_output.py:196",
                message="Synthesis exception handled without worker crash",
                data={"error": str(exc), "priority": msg.priority},
            )
            # endregion
            with self._lock:
                self._current_proc = None
                self._current_rank = None
            self._enabled = False
            self._disabled_reason = "synth_exception"
            return False
        with self._lock:
            self._current_proc = proc
        if proc is not None:
            try:
                proc.wait(timeout=30.0)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        with self._lock:
            self._current_proc = None
            self._current_rank = None
            self._last_spoken_ts[msg.signature] = time.monotonic()
        return True

    def _run(self) -> None:
        while not self._stop:
            try:
                if not self.process_next():
                    time.sleep(0.01)
            except Exception as exc:
                # region agent log
                _debug_log(
                    run_id=self._debug_run_id,
                    hypothesis_id="H4_worker_exception",
                    location="src/output/audio_output.py:167",
                    message="Audio worker loop crashed with exception",
                    data={"error": str(exc)},
                )
                # endregion
                raise

    def _spawn_piper(self, text: str) -> Any:
        ts = int(time.time() * 1000.0)
        out_wav = self.output_dir / f"tts_{ts}.wav"
        command = [
            self._piper_executable,
            "--model",
            str(self.model_path),
            "--output_file",
            str(out_wav),
        ]
        # region agent log
        _debug_log(
            run_id=self._debug_run_id,
            hypothesis_id="H1_piper_missing_or_not_in_path",
            location="src/output/audio_output.py:185",
            message="Preparing Piper process spawn",
            data={
                "model_path": str(self.model_path),
                "model_exists": self.model_path.exists(),
                "output_wav": str(out_wav),
                "command_0": command[0],
            },
        )
        # endregion
        return _piper_proc(command, text, run_id=self._debug_run_id)


def _piper_proc(command: list[str], text: str, *, run_id: str) -> Any:
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception as exc:
        # region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H2_piper_spawn_fails",
            location="src/output/audio_output.py:208",
            message="Piper process spawn failed",
            data={"error": str(exc), "command": command},
        )
        # endregion
        raise
    if proc.stdin is not None:
        try:
            proc.stdin.write(text)
            proc.stdin.close()
        except Exception:
            pass
    return proc
