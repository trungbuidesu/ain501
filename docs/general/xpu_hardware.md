# Tăng Tốc Phần Cứng: Intel XPU (Arc A770)

Dự án AIN501 được tối ưu hóa đặc biệt cho dòng card đồ họa chuyên dụng **Intel Arc A770 (16GB VRAM)** thông qua hệ sinh thái phần mềm Intel oneAPI.

## Môi Trường Tính Toán (Execution Environment)

Để tận dụng tối đa sức mạnh phần cứng, hệ thống yêu cầu các thành phần sau:

- **Intel oneAPI Base Toolkit**: Cung cấp các thư viện tối ưu hóa như `mkl`, `dnnl`.
- **Intel Extension for PyTorch (IPEX)**: Mở rộng khả năng của PyTorch để hỗ trợ thiết bị `xpu`.
- **oneDNN**: Thư viện nhân tính toán cho deep learning.

## Cấu Hình Phần Mềm (Software Stack)

```bash
# Kích hoạt môi trường (Windows)
conda activate ain501

# Import kiểm tra trong Python
import torch
import intel_extension_for_pytorch as ipex
print(torch.xpu.is_available()) # Kết quả phải là True
```

## Các Lưu Ý Kỹ Thuật (Technical Gotchas)

### 1. Dtype Compatibility
Intel XPU trên dòng Arc (Discrete Graphics) hỗ trợ tốt nhất `float32` và `bfloat16`. 
> [!WARNING] Lỗi XPUDoubleType
> Tránh sử dụng `float64` (Double) vì nhiều kernel trên XPU chưa hỗ trợ đầy đủ, dễ gây lỗi Runtime. Luôn ép kiểu về `.float()` trước khi đưa dữ liệu vào model.

### 2. Batch Size & Memory
Với 16GB VRAM, Arc A770 có thể xử lý tốt:
- **YOLOv8n**: Batch size up to 128 (imgsz 640).
- **MoViNet-A0**: Batch size 32-64 (resolution 172, 5-10 clips).

### 3. ONNX Runtime với OpenVINO
Đối với pha inference, khuyến khích sử dụng `OpenVINOExecutionProvider` để đạt latency thấp nhất trên hardware của Intel.

```python
providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)
```
