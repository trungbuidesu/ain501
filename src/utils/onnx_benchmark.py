"""Helpers for ONNX Runtime dry-run inference and latency benchmarks."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]


def get_process_rss_mb() -> float:
    """Return current process RSS in MiB (Windows-friendly)."""

    if os.name == "nt":
        return _windows_rss_mb()
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _windows_rss_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if ok:
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def get_nested_value(payload: Any, keys: list[str]) -> Any:
    """Traverse dict keys; return None if any step missing."""

    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_json_metric(path: Path, keys: list[str]) -> float | None:
    """Load a numeric metric from JSON at path using key path."""

    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = get_nested_value(data, keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numpy_dtype(name: str) -> Any:
    mapping: dict[str, Any] = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    return mapping.get(name, np.float32)


def feeds_from_input_specs(specs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build random numpy feeds from registry input_specs."""

    feeds: dict[str, np.ndarray] = {}
    for name, meta in specs.items():
        shape = list(meta["shape"])
        dtype_name = str(meta.get("dtype", "float32"))
        dt = numpy_dtype(dtype_name)
        if np.issubdtype(dt, np.integer):
            feeds[name] = np.random.randint(0, 1000, size=shape, dtype=dt)
        else:
            feeds[name] = np.random.randn(*shape).astype(dt)
    return feeds


def onnxruntime_version() -> str:
    if ort is None:
        return "not_installed"
    return str(ort.__version__)


def create_session(model_path: Path, providers: list[str] | None = None) -> Any:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    prov = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=prov)


def run_single_forward(session: Any, feeds: dict[str, np.ndarray]) -> list[Any]:
    input_names = [i.name for i in session.get_inputs()]
    ordered = {name: feeds[name] for name in input_names if name in feeds}
    if len(ordered) != len(input_names):
        raise ValueError(f"feeds missing for inputs: {input_names}")
    return session.run(None, ordered)


def benchmark_onnx(
    model_path: Path,
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Measure median/p95 latency (ms) and RSS delta for one ONNX model."""

    ram_before = get_process_rss_mb()
    session = create_session(model_path, providers=providers)
    ram_after_load = get_process_rss_mb()
    for _ in range(warmup):
        _ = run_single_forward(session, feeds)
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = run_single_forward(session, feeds)
        timings.append((time.perf_counter() - start) * 1000.0)
    ram_after = get_process_rss_mb()
    ordered = sorted(timings)
    p95_idx = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered))) - 1))
    return {
        "latency_ms_p50": float(statistics.median(timings)),
        "latency_ms_p95": float(ordered[p95_idx]),
        "ram_mb_before": ram_before,
        "ram_mb_after_load": ram_after_load,
        "ram_mb_after_run": ram_after,
        "ram_mb_delta": ram_after - ram_before,
        "file_size_mb": (
            model_path.stat().st_size / (1024.0 * 1024.0)
            if model_path.is_file()
            else 0.0
        ),
    }
