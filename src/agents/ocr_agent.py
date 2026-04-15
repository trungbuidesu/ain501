"""PaddleOCR-based text agent with coarse region proposals (Tier 1)."""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Windows + oneDNN: avoid NotImplementedError in some Paddle+PaddleX builds.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "false")


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One OCR line with axis-aligned bounding box in pixel coordinates."""

    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "position": {"x": self.x, "y": self.y, "w": self.w, "h": self.h},
            "confidence": self.confidence,
        }


def _bbox_from_poly(poly: Any) -> tuple[float, float, float, float] | None:
    if poly is None:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if arr.size < 4:
        return None
    if arr.ndim == 2 and arr.shape[1] >= 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
    return None


def _normalize_paddle_output(result: Any) -> list[dict[str, Any]]:
    if not result:
        return []
    if isinstance(result, list) and result and isinstance(result[0], list):
        page = result[0]
        lines_out: list[dict[str, Any]] = []
        for entry in page:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            box, rec = entry
            if isinstance(rec, (list, tuple)) and len(rec) == 2:
                text, score = rec[0], rec[1]
            else:
                continue
            lines_out.append({"box": box, "text": str(text), "score": float(score)})
        return lines_out
    if isinstance(result, list) and result:
        page = result[0]
        try:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            polys = page.get("rec_polys") or page.get("dt_polys")
        except (KeyError, TypeError):
            return []
        lines_out = []
        for i, text in enumerate(texts):
            poly = polys[i] if polys is not None and i < len(polys) else None
            sc = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            lines_out.append({"box": poly, "text": str(text), "score": sc})
        return lines_out
    return []


def _propose_rois(
    rgb: np.ndarray, *, grid: int = 2, pad: int = 8
) -> list[tuple[int, int, int, int]]:
    """Split frame into a coarse grid of ROIs to reduce full-frame OCR cost."""

    h, w = rgb.shape[:2]
    rois: list[tuple[int, int, int, int]] = []
    gh = max(1, h // grid)
    gw = max(1, w // grid)
    for gy in range(grid):
        for gx in range(grid):
            y1 = min(h - 1, gy * gh)
            x1 = min(w - 1, gx * gw)
            y2 = min(h, y1 + gh + pad)
            x2 = min(w, x1 + gw + pad)
            if y2 <= y1 or x2 <= x1:
                continue
            rois.append((y1, y2, x1, x2))
    return rois


def _edge_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 40, 120)


class OcrAgent:
    """Lazy PaddleOCR; runs on proposed regions and merges results."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        use_angle_cls: bool = True,
        lang: str = "en",
        grid: int = 2,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self._use_angle_cls = use_angle_cls
        self._lang = lang
        self._grid = int(grid)
        self._ocr: Any = None

    def _ensure_ocr(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "paddleocr is required for OcrAgent. Install paddlepaddle + paddleocr."
                ) from exc
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=self._use_angle_cls,
                    lang=self._lang,
                    show_log=False,
                )
            except TypeError:
                self._ocr = PaddleOCR(lang=self._lang)
        return self._ocr

    def _run_ocr(self, ocr: Any, crop: np.ndarray) -> Any:
        if hasattr(ocr, "ocr"):
            return ocr.ocr(crop, cls=self._use_angle_cls)
        return ocr.predict(crop)

    def read_text(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return filtered line dicts: text, position{x,y,w,h}, confidence."""

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must be HWC RGB uint8")
        ocr = self._ensure_ocr()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        edges = _edge_mask(gray)
        edge_density = float(np.mean(edges > 0))
        h, w = frame_rgb.shape[:2]
        merged: dict[tuple[int, int, str], OcrLine] = {}

        rois = _propose_rois(frame_rgb, grid=self._grid)
        for y1, y2, x1, x2 in rois:
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ce = float(np.mean(_edge_mask(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)) > 0))
            if ce < edge_density * 0.25 and edge_density > 0.02:
                continue
            result = self._run_ocr(ocr, crop)
            for line in _normalize_paddle_output(result):
                sc = float(line.get("score", 0.0))
                if sc < self.confidence_threshold:
                    continue
                bb = _bbox_from_poly(line.get("box"))
                if bb is None:
                    continue
                ox, oy, bw, bh = bb
                gx = x1 + ox
                gy = y1 + oy
                key = (int(gx // 8), int(gy // 8), str(line.get("text", "")))
                prev = merged.get(key)
                nl = OcrLine(
                    text=str(line.get("text", "")),
                    x=gx,
                    y=gy,
                    w=bw,
                    h=bh,
                    confidence=sc,
                )
                if prev is None or nl.confidence > prev.confidence:
                    merged[key] = nl

        if not merged and edge_density > 0.03:
            result = self._run_ocr(ocr, frame_rgb)
            for line in _normalize_paddle_output(result):
                sc = float(line.get("score", 0.0))
                if sc < self.confidence_threshold:
                    continue
                bb = _bbox_from_poly(line.get("box"))
                if bb is None:
                    continue
                ox, oy, bw, bh = bb
                merged[(0, 0, str(line.get("text", "")))] = OcrLine(
                    text=str(line.get("text", "")),
                    x=ox,
                    y=oy,
                    w=bw,
                    h=bh,
                    confidence=sc,
                )

        return [v.as_dict() for v in merged.values()]


def benchmark_ocr_agent_latency(
    agent: OcrAgent,
    frame_rgb: np.ndarray,
    *,
    samples: int = 20,
) -> dict[str, Any]:
    timings: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        _ = agent.read_text(frame_rgb)
        timings.append((time.perf_counter() - t0) * 1000.0)
    med = float(statistics.median(timings)) if timings else float("nan")
    return {
        "latency_ms_p50": med,
        "latency_ms_p95": _percentile_ms(timings, 95.0),
        "target_ms": 30.0,
        "meets_target": bool(med < 30.0),
    }


def _percentile_ms(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(len(s) - 1, lo + 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (hi - k) + s[hi] * (k - lo))
