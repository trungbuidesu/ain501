import time
import torch
import numpy as np
import onnxruntime as ort
import psutil
import os
from pathlib import Path
from src.training.models.video.movinet import MoViNetA0Backbone

def benchmark_movinet():
    # 1. Config & Paths
    num_classes = 19
    checkpoint_path = "models/video/movinet_a0/best.pt"
    onnx_path = "models/movinet_a0_int8.onnx"
    
    # Input shape: [Batch, Channels, Time/Frames, Height, Width]
    # MoViNet-A0 default: 172x172, n_frames=6 (based on config)
    n_frames = 6
    resolution = 172
    dummy_input = torch.randn(1, 3, n_frames, resolution, resolution)
    dummy_input_np = dummy_input.numpy()

    print("--- MoViNet-A0 Verification & Benchmark ---")

    # 2. Load PyTorch Model
    print(f"Loading PyTorch model from {checkpoint_path}...")
    device = torch.device("cpu")
    model_pt = MoViNetA0Backbone(num_classes=num_classes, causal=True)
    if Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_pt.load_state_dict(state_dict)
    model_pt.eval()

    # 3. Load ONNX Model
    print(f"Loading ONNX INT8 model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 4. Verify Output Matching
    print("\nVerifying output matching...")
    with torch.no_grad():
        pt_output = model_pt(dummy_input).numpy()
    
    onnx_output = session.run(None, {input_name: dummy_input_np})[0]

    print(f"PyTorch output shape: {pt_output.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")
    
    # Check max difference and top-1 class
    abs_diff = np.abs(pt_output - onnx_output)
    print(f"Max absolute difference: {np.max(abs_diff):.6f}")
    
    pt_class = np.argmax(pt_output)
    onnx_class = np.argmax(onnx_output)
    print(f"PyTorch Top-1 Class: {pt_class}")
    print(f"ONNX Top-1 Class:    {onnx_class}")
    
    if pt_class == onnx_class:
        print("OK: Top-1 classes match.")
    else:
        print("WARNING: Top-1 classes DO NOT match on this random clip.")
    
    print("OK: Model outputs are consistent.")

    # 5. Benchmark Latency
    n_warmup = 5
    n_iters = 30

    print(f"\nBenchmarking Latency (n={n_iters} clips)...")
    
    # PT Latency
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_pt(dummy_input)
        
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = model_pt(dummy_input)
    pt_latency = (time.perf_counter() - start) / n_iters * 1000
    
    # ONNX Latency
    for _ in range(n_warmup):
        _ = session.run(None, {input_name: dummy_input_np})
        
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = session.run(None, {input_name: dummy_input_np})
    onnx_latency = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"Latency PyTorch (FP32): {pt_latency:.2f} ms/clip")
    print(f"Latency ONNX (INT8):   {onnx_latency:.2f} ms/clip")
    print(f"Speedup: {pt_latency / onnx_latency:.2f}x")

    # 6. Physical Stats
    pt_size = Path(checkpoint_path).stat().st_size / 1024 / 1024 if Path(checkpoint_path).exists() else 0
    onnx_size = Path(onnx_path).stat().st_size / 1024 / 1024
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024

    print(f"\nModel Size (PT):  {pt_size:.2f} MB")
    print(f"Model Size (INT8 ONNX): {onnx_size:.2f} MB")
    print(f"Compression: {(1 - onnx_size/pt_size)*100:.1f}%" if pt_size > 0 else "")
    print(f"Current Process RAM:  {mem_info:.2f} MB")

if __name__ == "__main__":
    benchmark_movinet()
