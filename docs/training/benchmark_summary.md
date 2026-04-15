# Benchmark Summary (Preserved)

File này giữ các chỉ số benchmark chính trong repo để tránh mất thông tin khi `reports/` bị git-ignore hoặc chỉ lưu local.

## Key numbers (Week 6 baseline)

- **MobileNetV3-Small (FP32):** latency ~`3.2 ms`, RAM ~`480 MB`
- **YOLOv8n (INT8):** latency ~`24.8 ms`, `mAP50 = 0.51`
- **MoViNet-A0 (INT8):** latency ~`20.7 ms`, accuracy ~`52%`
- **MiniLM-L6 (ONNX):** latency ~`4.07 ms`, similarity gap ~`0.33`

## Notes

- Đây là baseline tham chiếu cho packaging/deploy decision.
- Khi benchmark chính thức được cập nhật, hãy đồng bộ lại file này cùng tài liệu training liên quan.
