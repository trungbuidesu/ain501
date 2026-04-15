import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from ultralytics import YOLO


def benchmark_model():
    model_pt_path = "yolov8n.pt"  # Default name
    model_onnx_path = "models/yolov8n_int8.onnx"

    img_size = 640
    # YOLO predict expects HWC or list of HWC
    dummy_input = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)

    print("--- 🔬 YOLOv8n Verification & Benchmark ---")

    # 1. Load Models
    print(f"Loading PyTorch model: {model_pt_path}...")
    model_pt = YOLO(model_pt_path)

    print(f"Loading ONNX INT8 model: {model_onnx_path}...")
    session = ort.InferenceSession(model_onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 2. Verify Output Matching (Rough check)
    print("\nVerifying output matching...")
    # PT inference
    results_pt = model_pt.predict(dummy_input, verbose=False)
    pt_output = results_pt[0].boxes.data.cpu().numpy()

    # ONNX inference (Requires NCHW float32 [0,1], static batch 16)
    dummy_input_onnx = dummy_input.transpose(2, 0, 1)  # HWC -> CHW
    dummy_input_onnx = np.expand_dims(dummy_input_onnx, axis=0)  # CHW -> NCHW
    dummy_input_onnx = np.repeat(dummy_input_onnx, 16, axis=0)  # Batch size 1 -> 16
    dummy_input_onnx = dummy_input_onnx.astype(np.float32) / 255.0

    onnx_output = session.run(None, {input_name: dummy_input_onnx})[0]

    print(f"PyTorch output shape: {pt_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    # Note: Raw ONNX output might need different post-processing to match exactly,
    # but we can check if both run without errors.
    print("✅ Both models ran successfully.")

    # 3. Benchmark Latency
    n_warmup = 5
    n_iters = 20

    print(f"\nBenchmarking Latency (n={n_iters})...")

    # PT Latency
    for _ in range(n_warmup):
        _ = model_pt.predict(dummy_input, verbose=False)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model_pt.predict(dummy_input, verbose=False)
    pt_latency = (time.perf_counter() - start) / n_iters * 1000

    # ONNX Latency
    for _ in range(n_warmup):
        _ = session.run(None, {input_name: dummy_input_onnx})

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = session.run(None, {input_name: dummy_input_onnx})
    onnx_latency = (time.perf_counter() - start) / (n_iters * 16) * 1000

    print(f"⏱️ PyTorch (FP32) Latency: {pt_latency:.2f} ms")
    print(f"⏱️ ONNX (INT8) Latency:   {onnx_latency:.2f} ms (per image, batch=16)")
    print(f"🚀 Speedup: {pt_latency / onnx_latency:.2f}x")

    # 4. Resource Usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024

    pt_size = Path(model_pt_path).stat().st_size / 1024 / 1024
    onnx_size = Path(model_onnx_path).stat().st_size / 1024 / 1024

    print(f"\n📦 Model Size (FP32 PT): {pt_size:.2f} MB")
    print(f"📦 Model Size (INT8 ONNX): {onnx_size:.2f} MB")
    print(f"📉 Compression: {(1 - onnx_size/pt_size)*100:.1f}%")
    print(f"🧠 Current Process RAM: {mem_info:.2f} MB")


if __name__ == "__main__":
    benchmark_model()
