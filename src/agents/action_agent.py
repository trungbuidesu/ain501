"""MoViNet-A0 ONNX action recognition for Tier 1 (temporal clip from ring buffer)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml  # type: ignore[import-untyped]

from src.capture.types import FramePacket
from src.training.data.core import DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD
from src.utils.onnx_benchmark import benchmark_onnx, create_session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_action_vocab() -> list[str]:
    cfg = _repo_root() / "configs" / "datasets" / "action_accessibility.yaml"
    if not cfg.is_file():
        return [
            "walk",
            "wave",
            "sit",
            "stand",
            "climb_stairs",
            "fall_floor",
            "run",
            "turn",
            "push",
            "shake_hands",
            "talk",
            "clap",
        ]
    payload = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    classes = payload.get("classes", {}) if isinstance(payload, dict) else {}
    hmdb = classes.get("hmdb51_accessibility")
    if isinstance(hmdb, list) and hmdb:
        return [str(x) for x in hmdb]
    return [
        "walk",
        "wave",
        "sit",
        "stand",
        "climb_stairs",
        "fall_floor",
        "run",
        "turn",
        "push",
        "shake_hands",
        "talk",
        "clap",
    ]


def resolve_movinet_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_absolute() else _repo_root() / p
    registry_path = _repo_root() / "models" / "registry.json"
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        for model in payload.get("models", []):
            if model.get("id") == "movinet_a0":
                artifacts = model.get("artifacts", {})
                int8_meta = artifacts.get("int8_onnx")
                if isinstance(int8_meta, dict) and int8_meta.get("path"):
                    return _repo_root() / str(int8_meta["path"])
    return _repo_root() / "models" / "movinet_a0_int8.onnx"


def _packets_to_clip_rgb(
    packets: list[FramePacket],
    *,
    target_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Stack frames to [T,H,W,C] uint8, sampling evenly if len != target."""

    if not packets:
        raise ValueError("packets must be non-empty")
    n = len(packets)
    if n >= target_frames:
        idxs = np.linspace(0, n - 1, target_frames, dtype=int)
        frames = [packets[int(i)].data for i in idxs]
    else:
        frames = [packets[i].data for i in range(n)]
        while len(frames) < target_frames:
            frames.append(frames[-1])
    out = [
        cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        for rgb in frames
    ]
    return np.stack(out, axis=0)


class ActionAgent:
    """MoViNet ONNX INT8: input [1,3,T,H,W] normalized like training."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        vocab: list[str] | None = None,
        input_frames: int = 6,
        height: int = 172,
        width: int = 172,
        providers: list[str] | None = None,
        session: Any | None = None,
        strict_artifacts: bool = True,
    ) -> None:
        resolved = resolve_movinet_path(model_path)
        self.model_path = resolved
        self.vocab = list(vocab or _default_action_vocab())
        self.input_frames = int(input_frames)
        self.height = int(height)
        self.width = int(width)
        if strict_artifacts and not self.model_path.is_file():
            raise FileNotFoundError(
                f"MoViNet ONNX missing: {self.model_path}. "
                "Train/export or set models.movinet_int8 in config."
            )
        self._session = session
        if session is None and self.model_path.is_file():
            self._session = create_session(self.model_path, providers=providers)
        if self._session is not None:
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            shape = list(getattr(inp, "shape", []) or [])
            # Align clip geometry with ONNX static dimensions when provided.
            if len(shape) >= 5:
                t_dim = shape[2]
                h_dim = shape[3]
                w_dim = shape[4]
                if isinstance(t_dim, int) and t_dim > 0:
                    self.input_frames = int(t_dim)
                if isinstance(h_dim, int) and h_dim > 0:
                    self.height = int(h_dim)
                if isinstance(w_dim, int) and w_dim > 0:
                    self.width = int(w_dim)
        else:
            self._input_name = "input"

        self._mean = np.array(DEFAULT_NORMALIZE_MEAN, dtype=np.float32).reshape(
            3, 1, 1, 1
        )
        self._std = np.array(DEFAULT_NORMALIZE_STD, dtype=np.float32).reshape(
            3, 1, 1, 1
        )

    def build_tensor(self, packets: list[FramePacket]) -> np.ndarray:
        """Build NCTHW float32 batch for ONNX runtime."""

        clip = _packets_to_clip_rgb(
            packets,
            target_frames=self.input_frames,
            height=self.height,
            width=self.width,
        )
        # [T,H,W,C] -> [C,T,H,W]
        x = np.transpose(clip.astype(np.float32) / 255.0, (3, 0, 1, 2))
        x = (x - self._mean) / self._std
        x = np.expand_dims(x, axis=0).astype(np.float32, copy=False)
        return x

    def predict(self, packets: list[FramePacket]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MoViNet session not available")
        if len(packets) < 1:
            raise ValueError("Need at least one frame packet")
        feed = self.build_tensor(packets)
        logits = self._session.run(None, {self._input_name: feed})[0]
        logits = np.asarray(logits, dtype=np.float32).ravel()
        idx = int(np.argmax(logits))
        ex = np.exp(logits - np.max(logits))
        probs = ex / np.sum(ex)
        topk = sorted(
            range(len(probs)),
            key=lambda i: float(probs[i]),
            reverse=True,
        )[:5]
        topk_out = [
            {
                "action": self.vocab[i] if i < len(self.vocab) else f"class_{i}",
                "confidence": float(probs[i]),
            }
            for i in topk
        ]
        label = self.vocab[idx] if idx < len(self.vocab) else f"class_{idx}"
        return {
            "action": label,
            "confidence": float(probs[idx]),
            "topk": topk_out,
        }

    def benchmark_cpu(
        self,
        *,
        warmup: int = 20,
        iterations: int = 100,
    ) -> dict[str, Any]:
        if not self.model_path.is_file():
            return {"error": "model_missing", "path": str(self.model_path)}
        feed = np.zeros(
            (1, 3, self.input_frames, self.height, self.width),
            dtype=np.float32,
        )
        return benchmark_onnx(
            self.model_path,
            {self._input_name: feed},
            warmup=warmup,
            iterations=iterations,
        )


def percentile_ms(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    xs = sorted(samples)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(len(xs) - 1, lo + 1)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def benchmark_action_agent_latency(
    agent: ActionAgent,
    *,
    samples: int = 50,
) -> dict[str, Any]:
    """Measure predict() latency on random frames (CPU)."""

    rng = np.random.default_rng(0)
    frames = [
        FramePacket(
            data=rng.integers(0, 255, size=(172, 172, 3), dtype=np.uint8),
            t_mono=0.0,
            source="file",
            frame_id=i,
        )
        for i in range(agent.input_frames)
    ]
    timings: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        _ = agent.predict(frames)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_ms_p50": float(statistics.median(timings)),
        "latency_ms_p95": percentile_ms(timings, 95.0),
        "target_ms": 15.0,
        "meets_target": bool(statistics.median(timings) < 15.0),
    }
