"""Draw bounding boxes and expert overlays on BGR frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.app.runtime import FrameProcessResult

_HAZARD = {"car", "truck", "bus", "motorcycle", "bicycle"}
_NAV = {"stairs", "traffic_light", "stop_sign"}


def _color_for_label(label: str) -> tuple[int, int, int]:
    low = label.lower()
    if low in _HAZARD:
        return (0, 0, 255)
    if low in _NAV:
        return (0, 255, 255)
    return (0, 255, 0)


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw YOLO-style instance boxes (BGR)."""

    out = frame_bgr.copy()
    for det in detections:
        label = str(det.get("label", ""))
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox") or {}
        try:
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        color = _color_for_label(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        pos = str(det.get("position", "") or "")
        dist = str(det.get("distance", "") or "")
        if pos or dist:
            side = f"{pos} {dist}".strip()
            cv2.putText(
                out,
                side[:40],
                (x1, min(y2 + 16, out.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def draw_action_overlay(
    frame_bgr: np.ndarray,
    frame_result: FrameProcessResult,
) -> np.ndarray:
    """Action label top-right."""

    ar = frame_result.agent_results or {}
    act = ar.get("action")
    if act is None or act.status != "ok" or not isinstance(act.payload, dict):
        return frame_bgr
    a = str(act.payload.get("action", "")).strip()
    if not a:
        return frame_bgr
    conf = float(act.payload.get("confidence", 0.0))
    h, w = frame_bgr.shape[:2]
    text = f"Action: {a} ({conf:.2f})"
    x = max(10, w - 320)
    cv2.rectangle(frame_bgr, (x - 4, 8), (w - 8, 36), (40, 40, 40), -1)
    cv2.putText(
        frame_bgr,
        text,
        (x, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr


def draw_ocr_overlay(
    frame_bgr: np.ndarray,
    frame_result: FrameProcessResult,
) -> np.ndarray:
    """OCR regions in cyan."""

    ar = frame_result.agent_results or {}
    oc = ar.get("ocr")
    if oc is None or oc.status != "ok" or not isinstance(oc.payload, list):
        return frame_bgr
    cyan = (255, 255, 0)
    for line in oc.payload:
        if not isinstance(line, dict):
            continue
        bb = line.get("bbox") or line.get("box")
        text = str(line.get("text", ""))[:80]
        if isinstance(bb, dict) and all(k in bb for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = int(bb["x1"]), int(bb["y1"])
            x2, y2 = int(bb["x2"]), int(bb["y2"])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cyan, 1)
            cv2.putText(
                frame_bgr,
                text,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                cyan,
                1,
                cv2.LINE_AA,
            )
    return frame_bgr


def draw_face_overlay(
    frame_bgr: np.ndarray,
    frame_result: FrameProcessResult,
) -> np.ndarray:
    """Face boxes in magenta."""

    ar = frame_result.agent_results or {}
    fp = ar.get("face_pose")
    if fp is None or fp.status != "ok" or not isinstance(fp.payload, dict):
        return frame_bgr
    mag = (255, 0, 255)
    for face in fp.payload.get("faces") or []:
        if not isinstance(face, dict):
            continue
        bb = face.get("bbox")
        em = str(face.get("emotion", ""))
        if isinstance(bb, dict) and all(k in bb for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = int(bb["x1"]), int(bb["y1"])
            x2, y2 = int(bb["x2"]), int(bb["y2"])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), mag, 2)
            cv2.putText(
                frame_bgr,
                em[:24],
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                mag,
                1,
                cv2.LINE_AA,
            )
    return frame_bgr


def compose_visual_overlay(
    frame_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    frame_result: FrameProcessResult,
) -> np.ndarray:
    """RGB in → BGR out with all overlays."""

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    bgr = draw_detections(bgr, detections)
    bgr = draw_action_overlay(bgr, frame_result)
    bgr = draw_ocr_overlay(bgr, frame_result)
    bgr = draw_face_overlay(bgr, frame_result)
    return bgr
