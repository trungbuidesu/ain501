# Mạng Video Biên Độ Thấp: MoViNet-A0 (Action Recognition)

Mô hình MoViNet-A0 là thành phần cốt lõi trong hệ thống nhận diện hành động thời gian thực của dự án AIN501. Với kiến trúc được tối ưu hóa cho thiết bị di động và biên, MoViNet cho phép xử lý video dạng luồng (streaming) với độ trễ cực thấp.

## Kiến Trúc Đặc Thù (Advanced Architecture)

Khác với các mạng 3D CNN truyền thống, MoViNet-A0 sở hữu các đặc tính kỹ thuật độc bản:

- **Causal Convolutions**: Các lớp tích chập chỉ nhìn về quá khứ, loại bỏ nhu cầu nhìn trước tương lai (future frames), giúp model có thể output kết quả ngay khi nhận được khung hình mới.
- **Stream Buffers**: Cơ chế lưu trữ trạng thái trung gian (activations) giúp duy trì tính liên tục của hành động mà không cần xử lý lại toàn bộ clip.
- **Temporal Squeeze-and-Excitation (SE)**: Tối ưu hóa trọng số các đặc trưng theo thời gian, tăng cường khả năng nhận diện các hành động có biến thiên nhanh.

## Tiến Trình Huấn Luyện (Fine-Tuning Process)

Mô hình được huấn luyện trên hạ tầng **Intel XPU (Arc A770)** với các thông số:
- **Tập dữ liệu**: UCF-101 (subset 19 lớp hỗ trợ tiếp cận).
- **Độ phân giải**: 172x172 (Native resolution).
- **Kết quả**: Đạt **Top-1 Accuracy ~62.6%** sau 50 epochs.

## Lượng Tử Hóa và Xuất Bản (PTQ & Deployment)

Để tối ưu hóa cho môi trường thực tế, mô hình đã được xử lý qua pipeline:

1. **Monkey-patching**: Thay thế `AdaptiveAvgPool3d` bằng `AvgPool3d` để đảm bảo tính tương thích tuyệt đối với ONNX Opsets.
2. **PTQ (Post-Training Quantization)**: 
    - Chuyển đổi sang định dạng **INT8** sử dụng ONNX Runtime.
    - Kích thước mô hình giảm từ **10.4MB** xuống còn **4.42MB**.
    - Duy trì độ chính xác cao nhờ bước Calibration trên dữ liệu thực.

> [!TIP] Hiệu Năng Thực Tế
> Mô hình INT8 ONNX đạt tốc độ xử lý vượt trội trên CPU và GPU Intel thông qua OpenVINO, đảm bảo trải nghiệm streaming 30+ FPS.
