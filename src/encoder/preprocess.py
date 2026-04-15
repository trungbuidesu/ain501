"""Preprocessing helpers for ONNX visual encoders."""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

IMAGENET_MEAN: Final[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: Final[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_mobilenet_frame(
    frame_rgb: np.ndarray,
    *,
    image_size: int = 224,
) -> np.ndarray:
    """Resize + normalize RGB uint8 frame to NCHW float32 batch tensor."""

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must be HWC RGB")
    resized = cv2.resize(
        frame_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32, copy=False)
