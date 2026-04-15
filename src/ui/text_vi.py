"""Vietnamese-capable text drawing (OpenCV putText is ASCII-only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def put_vietnamese_lines(
    frame_bgr: np.ndarray,
    lines: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]],
) -> np.ndarray:
    """Draw multiple lines using PIL. Each tuple: text, (x,y), font_size, BGR color."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        for text, pos, _fs, col in lines:
            cv2.putText(
                frame_bgr,
                text.encode("ascii", errors="replace").decode(),
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )
        return frame_bgr

    bgr = frame_bgr.copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font_path = _pick_font()
    for text, pos, fs, col_bgr in lines:
        col_rgb = (int(col_bgr[2]), int(col_bgr[1]), int(col_bgr[0]))
        try:
            if font_path:
                font = ImageFont.truetype(str(font_path), fs)
            else:
                font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=col_rgb)
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out


def _pick_font() -> Path | None:
    windir = __import__("os").environ.get("WINDIR", "C:\\Windows")
    arial = Path(windir) / "Fonts" / "arial.ttf"
    if arial.is_file():
        return arial
    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if dejavu.is_file():
        return dejavu
    return None


def overlay_alpha_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> None:
    """In-place semi-transparent rectangle."""

    x1, x2 = sorted((max(0, x1), min(frame.shape[1] - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(frame.shape[0] - 1, y2)))
    roi = frame[y1 : y2 + 1, x1 : x2 + 1]
    if roi.size == 0:
        return
    solid = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(solid, alpha, roi, 1.0 - alpha, 0, roi)
