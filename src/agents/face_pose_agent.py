"""MediaPipe Face + Pose (Tasks API) with rule-based emotion/pose labels."""

from __future__ import annotations

import statistics
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_task_model(model_dir: Path, url: str, filename: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    if not path.is_file():
        urllib.request.urlretrieve(url, path)
    return path


def _face_emotion_rule(
    landmarks: Sequence[tuple[float, float, float]],
) -> tuple[str, float]:
    """Map simple mesh ratios to coarse emotion labels."""

    def d(i: int, j: int) -> float:
        xi, yi, _ = landmarks[i]
        xj, yj, _ = landmarks[j]
        return float(np.hypot(xi - xj, yi - yj))

    face_w = d(33, 263)
    mouth_open = d(13, 14) / max(face_w, 1e-6)
    mouth_width = d(61, 291) / max(face_w, 1e-6)
    eye_open_l = d(159, 145) / max(face_w, 1e-6)
    eye_open_r = d(386, 374) / max(face_w, 1e-6)

    if mouth_open > 0.18:
        return "surprised", min(1.0, mouth_open * 3.0)
    if mouth_width > 0.52:
        return "happy", min(1.0, mouth_width * 1.2)
    if eye_open_l < 0.04 and eye_open_r < 0.04:
        return "sad", 0.55
    return "neutral", 0.6


def _pose_rule(lm_list: Sequence[Any]) -> tuple[str, float]:
    if len(lm_list) < 33:
        return "standing", 0.3

    def yv(idx: int) -> tuple[float, float]:
        m = lm_list[idx]
        y = float(m.y or 0.0)
        v = float(m.visibility) if m.visibility is not None else 1.0
        return y, v

    ls_y, _ = yv(11)
    rs_y, _ = yv(12)
    lh_y, _ = yv(23)
    rh_y, _ = yv(24)
    lw_y, lw_v = yv(15)
    rw_y, rw_v = yv(16)
    sh_y = (ls_y + rs_y) / 2.0
    hip_y = (lh_y + rh_y) / 2.0
    torso = float(hip_y - sh_y)
    arms_up = lw_v > 0.5 and rw_v > 0.5 and lw_y < ls_y - 0.03 and rw_y < rs_y - 0.03
    if arms_up:
        return "arms_raised", 0.75
    if torso < 0.08:
        return "sitting", 0.65
    return "standing", 0.7


class FacePoseAgent:
    """Lazy MediaPipe FaceLandmarker + PoseLandmarker."""

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        strict_artifacts: bool = False,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self._model_dir = (
            Path(model_dir)
            if model_dir
            else root / "reports" / "prebuilt_week5" / "_mediapipe_models"
        )
        self._strict = strict_artifacts
        self._face_lm: Any = None
        self._pose_lm: Any = None
        self._mp: Any = None

    def _ensure(self) -> None:
        if self._face_lm is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import FaceLandmarker, PoseLandmarker
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for FacePoseAgent") from exc
        self._mp = mp
        try:
            face_path = _ensure_task_model(
                self._model_dir, FACE_LANDMARKER_URL, "face_landmarker.task"
            )
            pose_path = _ensure_task_model(
                self._model_dir, POSE_LANDMARKER_LITE_URL, "pose_landmarker_lite.task"
            )
        except OSError as exc:
            if self._strict:
                raise FileNotFoundError(
                    f"Could not download MediaPipe task models into {self._model_dir}"
                ) from exc
            raise
        self._face_lm = FaceLandmarker.create_from_model_path(str(face_path))
        self._pose_lm = PoseLandmarker.create_from_model_path(str(pose_path))

    def analyze(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        """Return structured face/pose payload."""

        self._ensure()
        mp = self._mp
        assert mp is not None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        fres = self._face_lm.detect(mp_image)
        pres = self._pose_lm.detect(mp_image)

        faces_out: list[dict[str, Any]] = []
        if fres.face_landmarks:
            h, w = frame_rgb.shape[:2]
            for lms in fres.face_landmarks:
                tuples = [
                    (float(p.x or 0.0), float(p.y or 0.0), float(p.z or 0.0))
                    for p in lms
                ]
                xs = [t[0] for t in tuples]
                ys = [t[1] for t in tuples]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                emotion, escore = _face_emotion_rule(tuples)
                faces_out.append(
                    {
                        "emotion": emotion,
                        "bbox": {
                            "x": float(x1 * w),
                            "y": float(y1 * h),
                            "w": float((x2 - x1) * w),
                            "h": float((y2 - y1) * h),
                        },
                        "score": float(escore),
                    }
                )

        poses_out: list[dict[str, Any]] = []
        if pres.pose_landmarks:
            for plm in pres.pose_landmarks:
                ptype, pscore = _pose_rule(plm)
                ys = [float(m.y or 0.0) for m in plm]
                xs = [float(m.x or 0.0) for m in plm]
                h, w = frame_rgb.shape[:2]
                poses_out.append(
                    {
                        "pose_type": ptype,
                        "bbox": {
                            "x": float(min(xs) * w),
                            "y": float(min(ys) * h),
                            "w": float((max(xs) - min(xs)) * w),
                            "h": float((max(ys) - min(ys)) * h),
                        },
                        "score": float(pscore),
                    }
                )

        return {"faces": faces_out, "poses": poses_out}


def benchmark_face_pose_latency(
    agent: FacePoseAgent,
    frame_rgb: np.ndarray,
    *,
    samples: int = 30,
) -> dict[str, Any]:
    timings: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        _ = agent.analyze(frame_rgb)
        timings.append((time.perf_counter() - t0) * 1000.0)
    med = float(statistics.median(timings)) if timings else float("nan")
    xs = sorted(timings)
    p95 = xs[min(len(xs) - 1, max(0, int(round(0.95 * len(xs))) - 1))]
    return {
        "latency_ms_p50": med,
        "latency_ms_p95": float(p95),
        "target_ms": 15.0,
        "meets_target": bool(med < 15.0),
    }
