"""MobileNetV3-Small ONNX visual feature extractor with LRU cache."""

from __future__ import annotations

import json
import statistics
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from src.capture.types import FramePacket
from src.encoder.preprocess import preprocess_mobilenet_frame
from src.utils.onnx_benchmark import create_session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path() -> Path:
    registry_path = _repo_root() / "models" / "registry.json"
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        for model in payload.get("models", []):
            if model.get("id") == "mobilenetv3_small":
                artifacts = model.get("artifacts", {})
                fp32 = artifacts.get("fp32_onnx")
                if isinstance(fp32, dict) and fp32.get("path"):
                    return _repo_root() / str(fp32["path"])
    return _repo_root() / "models" / "mobilenetv3_small_fp32.onnx"


def _packet_cache_key(packet: FramePacket) -> tuple[str, int, str | None]:
    return packet.source, packet.frame_id, packet.path


class VisualEncoder:
    """ONNX Runtime visual encoder for frame-level embedding extraction."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        image_size: int = 224,
        cache_size: int = 256,
        enable_cache: bool = True,
        providers: list[str] | None = None,
        session: Any | None = None,
        output_index: int = 0,
    ) -> None:
        self.model_path = (
            Path(model_path) if model_path is not None else _default_model_path()
        )
        self.image_size = int(image_size)
        self.cache_size = int(cache_size)
        self.enable_cache = bool(enable_cache)
        self.output_index = int(output_index)
        self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        self._session = session
        if self._session is None:
            self._session = create_session(self.model_path, providers=providers)
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("Encoder ONNX has no inputs")
        self._input_name = inputs[0].name

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def _get_cached(self, key: tuple[Any, ...]) -> np.ndarray | None:
        if not self.enable_cache:
            return None
        value = self._cache.get(key)
        if value is None:
            self._cache_misses += 1
            return None
        self._cache.move_to_end(key)
        self._cache_hits += 1
        return value.copy()

    def _put_cache(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        if not self.enable_cache:
            return
        self._cache[key] = value.copy()
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def encode(
        self,
        frame_rgb: np.ndarray,
        *,
        cache_key: tuple[Any, ...] | None = None,
    ) -> np.ndarray:
        """Encode RGB frame into a flattened feature vector."""

        key = cache_key
        if key is not None:
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        x = preprocess_mobilenet_frame(frame_rgb, image_size=self.image_size)
        outputs = self._session.run(None, {self._input_name: x})
        if self.output_index >= len(outputs):
            raise IndexError("output_index out of range")
        vec = (
            np.asarray(outputs[self.output_index])
            .reshape(-1)
            .astype(np.float32, copy=False)
        )

        if key is not None:
            self._put_cache(key, vec)
        return vec

    def encode_packet(self, packet: FramePacket) -> np.ndarray:
        return self.encode(packet.data, cache_key=_packet_cache_key(packet))

    def benchmark_latency(
        self,
        frames: list[np.ndarray],
        *,
        warmup: int = 3,
        iterations: int = 20,
    ) -> dict[str, float]:
        if not frames:
            raise ValueError("frames must not be empty")
        for i in range(warmup):
            _ = self.encode(frames[i % len(frames)], cache_key=None)
        samples: list[float] = []
        for i in range(iterations):
            frame = frames[i % len(frames)]
            t0 = time.perf_counter()
            _ = self.encode(frame, cache_key=None)
            samples.append((time.perf_counter() - t0) * 1000.0)
        ordered = sorted(samples)
        p95_idx = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered))) - 1))
        return {
            "latency_ms_p50": float(statistics.median(samples)),
            "latency_ms_p95": float(ordered[p95_idx]),
        }

    def verify_latency_against_phase0(
        self,
        measured_p50_ms: float,
        *,
        benchmark_report_path: Path | str = Path("reports/week6/benchmark_report.json"),
        tolerance_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Compare measured p50 latency against Phase 0 benchmark row."""

        report_path = Path(benchmark_report_path)
        benchmark_p50 = None
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            for row in payload.get("rows", []):
                if row.get("model_id") == "mobilenetv3_small":
                    benchmark_p50 = row.get("fp32_latency_ms")
                    break
        if benchmark_p50 is None:
            return {
                "status": "inconclusive",
                "reason": "phase0_mobilenet_row_missing",
                "measured_p50_ms": float(measured_p50_ms),
            }
        allowed_upper = float(benchmark_p50) * (1.0 + float(tolerance_ratio))
        return {
            "status": "ok" if float(measured_p50_ms) <= allowed_upper else "warn",
            "benchmark_p50_ms": float(benchmark_p50),
            "measured_p50_ms": float(measured_p50_ms),
            "allowed_upper_ms": allowed_upper,
        }
