import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_isolated():
    model_path = "models/yolov8n_int8.onnx"
    img_size = 640
    batch_size = 16  # Model has static batch 16

    print("--- Isolated YOLOv8n INT8 Benchmark ---")

    ram_start = get_ram_mb()
    print(f"RAM before loading: {ram_start:.2f} MB")

    # 1. Load Model
    start_load = time.perf_counter()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    load_time = (time.perf_counter() - start_load) * 1000

    ram_after_load = get_ram_mb()
    print(f"RAM after loading:  {ram_after_load:.2f} MB")
    print(f"Model Load Time:    {load_time:.2f} ms")

    # 2. Prepare Dummy Input
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)

    # 3. Warmup
    print("Warmup...")
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})

    # 4. Benchmark Latency
    n_iters = 30
    print(f"Benchmarking (n={n_iters})...")
    start_bench = time.perf_counter()
    for _ in range(n_iters):
        _ = session.run(None, {input_name: dummy_input})
    total_time = (time.perf_counter() - start_bench) * 1000

    latency_per_batch = total_time / n_iters
    latency_per_img = latency_per_batch / batch_size

    ram_final = get_ram_mb()

    print("\n--- Final Results ---")
    print(f"Latency per Batch ({batch_size}): {latency_per_batch:.2f} ms")
    print(f"Latency per Image:     {latency_per_img:.2f} ms")
    print(f"RAM Start:             {ram_start:.2f} MB")
    print(f"RAM Final:             {ram_final:.2f} MB")
    print(f"RAM Delta (Net):       {ram_final - ram_start:.2f} MB")
    print(
        f"Model Size on Disk:    {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB"
    )


if __name__ == "__main__":
    benchmark_isolated()
