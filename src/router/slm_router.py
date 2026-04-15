"""ONNX scene classifier router with heuristic fallback when confidence is low."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.utils.onnx_benchmark import create_session

DEFAULT_SCENE_NAMES: tuple[str, ...] = (
    "static",
    "motion",
    "text_present",
    "face_detected",
    "scene_change",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_scene_classifier_path(explicit: Path | str | None = None) -> Path:
    """Resolve INT8 scene classifier ONNX from registry or default path."""

    if explicit is not None:
        p = Path(explicit)
        return p if p.is_absolute() else _repo_root() / p
    registry_path = _repo_root() / "models" / "registry.json"
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        for model in payload.get("models", []):
            if model.get("id") == "scene_classifier":
                artifacts = model.get("artifacts", {})
                int8_meta = artifacts.get("int8_onnx")
                if isinstance(int8_meta, dict) and int8_meta.get("path"):
                    return _repo_root() / str(int8_meta["path"])
    return _repo_root() / "models" / "scene_classifier_int8.onnx"


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@dataclass(frozen=True, slots=True)
class FallbackSignals:
    """Heuristic signals for routing when the model confidence is low."""

    motion_diff: float
    edge_density: float
    skin_ratio: float


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Resolved scene and which agents should run."""

    scene_type: str
    confidence: float
    active_agents: frozenset[str]
    used_router_model: bool
    used_fallback: bool
    signals: FallbackSignals
    raw_probs: tuple[float, ...] | None


def compute_fallback_signals(
    current_rgb: np.ndarray,
    previous_rgb: np.ndarray | None,
    *,
    skin_hsv_lower: tuple[int, int, int] = (0, 48, 80),
    skin_hsv_upper: tuple[int, int, int] = (25, 255, 255),
) -> FallbackSignals:
    """Compute motion diff, edge density, and skin-like pixel ratio."""

    if current_rgb.ndim != 3 or current_rgb.shape[2] != 3:
        raise ValueError("current_rgb must be HWC RGB uint8")
    gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    if previous_rgb is not None:
        pg = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        motion_diff = float(np.mean(np.abs(gray.astype(np.float32) - pg.astype(np.float32))))
    else:
        motion_diff = 0.0

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    hsv = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)
    lo = np.array(skin_hsv_lower, dtype=np.uint8)
    hi = np.array(skin_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    skin_ratio = float(np.mean(mask > 0))

    return FallbackSignals(
        motion_diff=motion_diff,
        edge_density=edge_density,
        skin_ratio=skin_ratio,
    )


def _infer_scene_from_signals(
    signals: FallbackSignals,
    *,
    motion_diff_threshold: float,
    edge_density_threshold: float,
    skin_ratio_threshold: float,
    motion_diff_scene_change: float,
) -> str:
    """Map signals to a scene label when router confidence is low."""

    if signals.motion_diff >= motion_diff_scene_change:
        return "scene_change"
    if signals.motion_diff >= motion_diff_threshold:
        return "motion"
    if signals.edge_density >= edge_density_threshold:
        return "text_present"
    if signals.skin_ratio >= skin_ratio_threshold:
        return "face_detected"
    return "static"


def _normalize_scene_mapping(
    raw: Mapping[str, Any],
    default_agents: frozenset[str],
) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for scene in DEFAULT_SCENE_NAMES:
        key = str(scene)
        if key in raw:
            agents = raw[key]
            if isinstance(agents, (list, tuple, set, frozenset)):
                out[key] = frozenset(str(a) for a in agents)
            else:
                out[key] = default_agents
        else:
            out[key] = default_agents
    return out


class SLMRouter:
    """Loads scene ONNX, runs softmax, applies confidence gating and fallback."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        class_names: tuple[str, ...] = DEFAULT_SCENE_NAMES,
        scene_to_agents: Mapping[str, Any] | None = None,
        confidence_threshold: float = 0.45,
        image_size: int = 160,
        fallback: Mapping[str, Any] | None = None,
        providers: list[str] | None = None,
        session: Any | None = None,
        strict_artifacts: bool = True,
    ) -> None:
        self._root = _repo_root()
        self.class_names = tuple(class_names)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        fb = dict(fallback or {})
        self._motion_diff_thr = float(fb.get("motion_diff_threshold", 8.0))
        self._edge_density_thr = float(fb.get("edge_density_threshold", 0.08))
        self._skin_ratio_thr = float(fb.get("skin_ratio_threshold", 0.12))
        self._motion_scene_change_thr = float(fb.get("motion_diff_scene_change", 18.0))

        default_agents = frozenset({"object"})
        self._scene_to_agents = _normalize_scene_mapping(
            dict(scene_to_agents or {}),
            default_agents,
        )

        resolved = resolve_scene_classifier_path(model_path)
        self.model_path = resolved
        if strict_artifacts and not self.model_path.is_file():
            raise FileNotFoundError(
                f"Scene classifier ONNX missing: {self.model_path}. "
                "Train/export or set router.model_path to a valid file."
            )
        self._session = session
        if session is None and self.model_path.is_file():
            self._session = create_session(self.model_path, providers=providers)
        if self._session is not None:
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
        else:
            self._input_name = "images"

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        resized = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = (x - self._mean) / self._std
        return x.astype(np.float32, copy=False)

    def _run_model(self, x: np.ndarray) -> tuple[np.ndarray, tuple[float, ...], int]:
        if self._session is None:
            raise RuntimeError("ONNX session not available (missing model or strict mode off)")
        logits = self._session.run(None, {self._input_name: x})[0]
        probs = _softmax(np.asarray(logits, dtype=np.float32))
        idx = int(np.argmax(probs))
        return probs, tuple(float(p) for p in probs.ravel()), idx

    def route(
        self,
        frame_rgb: np.ndarray,
        previous_rgb: np.ndarray | None,
    ) -> RoutingDecision:
        signals = compute_fallback_signals(frame_rgb, previous_rgb)

        used_model = False
        used_fallback = False
        raw_probs: tuple[float, ...] | None = None
        scene_idx = 0
        probs: np.ndarray | None = None

        if self._session is not None:
            x = self._preprocess(frame_rgb)
            probs, raw_probs, scene_idx = self._run_model(x)
            used_model = True
            conf = float(probs.ravel()[scene_idx])
            if conf < self.confidence_threshold:
                used_fallback = True
                fb_scene = _infer_scene_from_signals(
                    signals,
                    motion_diff_threshold=self._motion_diff_thr,
                    edge_density_threshold=self._edge_density_thr,
                    skin_ratio_threshold=self._skin_ratio_thr,
                    motion_diff_scene_change=self._motion_scene_change_thr,
                )
                scene_type = fb_scene
                confidence = conf
            else:
                scene_type = self.class_names[scene_idx]
                confidence = conf
        else:
            used_fallback = True
            scene_type = _infer_scene_from_signals(
                signals,
                motion_diff_threshold=self._motion_diff_thr,
                edge_density_threshold=self._edge_density_thr,
                skin_ratio_threshold=self._skin_ratio_thr,
                motion_diff_scene_change=self._motion_scene_change_thr,
            )
            confidence = 0.0

        active = self._scene_to_agents.get(scene_type, frozenset({"object"}))
        return RoutingDecision(
            scene_type=scene_type,
            confidence=confidence,
            active_agents=active,
            used_router_model=used_model,
            used_fallback=used_fallback,
            signals=signals,
            raw_probs=raw_probs,
        )
