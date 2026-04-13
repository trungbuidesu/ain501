# Intel Arc A770 — XPU Notes

> **GPU đã được xác nhận hoạt động** với PyTorch training trên hệ thống này.

## Thông tin phần cứng
- **GPU**: Intel(R) Arc(TM) A770 Graphics
- **Backend**: XPU (Intel oneAPI / SYCL)
- **PyTorch build**: `torch 2.11.0+xpu`

## Cài đặt PyTorch cho Intel XPU

> [!IMPORTANT]
> **KHÔNG** dùng `pip install torch` mặc định — nó chỉ cài bản CPU.
> Phải cài từ index XPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Các package Intel runtime được cài kèm tự động:
- `intel-sycl-rt`, `intel-opencl-rt`, `dpcpp-cpp-rt`
- `onemkl-sycl-*` (BLAS, DFT, LAPACK, RNG, Sparse)
- `triton-xpu`

## Sử dụng XPU trong code

### Chuyển model/tensor sang XPU
```python
import torch

device = torch.device("xpu")

# Model
model = model.to(device)

# Tensor
x = torch.randn(32, 64, device=device)
```

### Kiểm tra XPU availability
```python
import torch
print(torch.xpu.is_available())       # True
print(torch.xpu.device_count())       # 1
print(torch.xpu.get_device_name(0))   # Intel(R) Arc(TM) A770 Graphics
```

## Lưu ý quan trọng

### ⚠️ Resizable BAR
- **BẮT BUỘC** bật **Resizable BAR** trong BIOS.
- Nếu không bật, performance sẽ rất kém hoặc GPU có thể không được nhận.

### ⚠️ Driver
- Luôn cập nhật Intel Arc GPU driver mới nhất.
- Download tại: https://www.intel.com/content/www/us/en/download-center/home.html

### ⚠️ iGPU conflict
- Nếu có iGPU (Intel UHD), có thể cần disable trong BIOS để tránh conflict.
- Hoặc chỉ định rõ device index (`torch.device("xpu:0")`).

### ⚠️ Tương thích thư viện
- **CUDA-only libraries sẽ KHÔNG hoạt động** trên XPU (vd: một số custom CUDA kernels).
- Kiểm tra từng thư viện xem có hỗ trợ XPU/CPU fallback không.
- `ultralytics` (YOLOv8) hỗ trợ XPU natively.

### ⚠️ Mixed Precision
```python
# XPU hỗ trợ autocast tương tự CUDA
with torch.autocast(device_type="xpu", dtype=torch.float16):
    output = model(input)
```

## Kết quả sanity check

| Metric | Kết quả |
|--------|---------|
| XPU detected | ✅ |
| Device name | Intel(R) Arc(TM) A770 Graphics |
| 20-epoch training | 0.349s |
| Final loss | 1.9693 |
| Status | **PASSED** ✅ |

## Quy tắc khi viết code

1. **Dùng `"xpu"` thay cho `"cuda"`** trong device selection.
2. **Fallback pattern** nên là: `xpu` → `cpu` (không qua `cuda`).
3. Code mẫu device selection:
   ```python
   device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
   ```
4. **Không hardcode** `"cuda"` bất kỳ đâu trong project.
