"""Post-training quantization for YOLO ONNX using real image calibration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def _preprocess_image(path: Path, input_size: int) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        img_rgb,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32, copy=False)


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        image_paths: list[Path],
        input_size: int,
    ) -> None:
        self._input_name = input_name
        self._image_paths = image_paths
        self._input_size = input_size
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._image_paths):
            p = self._image_paths[self._idx]
            self._idx += 1
            try:
                x = _preprocess_image(p, self._input_size)
            except Exception:
                continue
            return {self._input_name: x}
        return None


def _benchmark(model_path: Path, iterations: int = 40) -> dict[str, float]:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(5):
        sess.run(None, {input_name: x})
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "p50_ms": float(times[len(times) // 2]),
        "p95_ms": float(times[int(len(times) * 0.95) - 1]),
        "min_ms": float(times[0]),
        "max_ms": float(times[-1]),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-input",
        type=Path,
        default=Path("models/yolov8n.onnx"),
    )
    p.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/yolov8n_int8_calib.onnx"),
    )
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("data/external/coco2017/val2017"),
    )
    p.add_argument("--num-images", type=int, default=800)
    p.add_argument("--input-size", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model_input = args.model_input.resolve()
    model_output = args.model_output.resolve()
    calib_dir = args.calib_dir.resolve()
    if not model_input.is_file():
        raise FileNotFoundError(f"model input not found: {model_input}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"calibration directory not found: {calib_dir}")

    image_paths = sorted(calib_dir.glob("*.jpg"))[: int(args.num_images)]
    if not image_paths:
        raise ValueError(f"no calibration images in {calib_dir}")

    sess = ort.InferenceSession(str(model_input), providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(getattr(input_meta, "shape", []))
    if input_shape and isinstance(input_shape[0], int) and input_shape[0] != 1:
        raise ValueError(f"expect batch=1 model for PTQ, got shape={input_shape}")

    reader = ImageCalibrationReader(
        input_name=input_name,
        image_paths=image_paths,
        input_size=int(args.input_size),
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )

    fp32_mb = model_input.stat().st_size / (1024 * 1024)
    int8_mb = model_output.stat().st_size / (1024 * 1024)
    bench = _benchmark(model_output)
    print(f"[ok] calib_images={len(image_paths)}")
    print(f"[ok] fp32_model={model_input} ({fp32_mb:.2f} MB)")
    print(f"[ok] int8_model={model_output} ({int8_mb:.2f} MB)")
    print(
        "[ok] int8_benchmark_ms "
        f"p50={bench['p50_ms']:.2f} p95={bench['p95_ms']:.2f} "
        f"min={bench['min_ms']:.2f} max={bench['max_ms']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
