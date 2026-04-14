# Định Danh Hệ Tiêu Biểu: YOLOv8n (Object Detection Backbone)

Mạng Nơ-ron Đa Quy YOLOv8n (Nano) là trái tim vận hành hệ sinh thái Thị giác Không Gian nhận diện thực thể (Object Detection Module) phục vụ ứng dụng ranh giới Hỗ trợ Tiếp cận. Mô hình này được cấp quyền thông dịch trực tiếp toàn bộ chuỗi đa biến thể COCO-Accessibility, qua đó đưa ra nhận thức nhanh và chính xác nhất cho chu kỳ sống của thiết bị.

## Cấu Trúc Khối Mạng Cực Rễ (Core Architecture Anatomy)

Kế thừa phương thức siêu cấp biểu đồ (Graph evolution), kiến trúc YOLOv8n phân mảnh tinh xảo thành các phân tầng cục bộ:

- **Bộ máy Chiết xuất Vùng Tối (CSPDarknet Backbone):** Kiến trúc rút trích tinh quang đa nhánh `C2f` (Cross Stage Partial feature block) với hệ số nhân học tham số siêu nhẹ.
- **Tuyển Ghép Thông Cổ (PANet-FPN Neck Bridge):** Phức hợp mạng hòa nhập thông phân luồng hai tuyến giao (Top-down và Bottom-up spatial feature fusions) tăng cường nhận diện với đa định quy kích thước (Multi-scale object parsing).
- **Trục Đầu Phân Rã Khối (Decoupled Anchor-free Head):** Điểm chóp xé tách tuyệt đối khối Phân loại ranh lớp (Classification logic), Định vị tọa độ (Box regression regression loss) và Hệ biểu đồ đặc tính tụ góc (DFL - Distribution Focal Loss parameters). 

```mermaid
graph TD
    A[Giao Ước Ảnh Ngõ Vào <br/> RGB Image 640x640] -->|Chuẩn Hóa Normalize| B(Khối Nhân CSPDarknet <br/> C2f Backbone Feature Extraction)
    B --> C[Phân Biểu Chéo Không Gian <br/> PAN-FPN Neck]
    C -->|Mảng Chiết Trích 1| D(Decoupled Head - Scale 1)
    C -->|Mảng Chiết Trích 2| E(Decoupled Head - Scale 2)
    C -->|Mảng Chiết Trích 3| F(Decoupled Head - Scale 3)
    D & E & F --> G{Tuyến Định Lượng Loại Trừ NMS <br/> Non-Maximum Suppression}
    G --> H[Bbox Tọa Độ, Dải Nhận Diện Lớp <br/> Output: Class + Limits]
```

## Khuyếch Đại Năng Trí Học Sâu (Supervised Fine-Tuning Strategy)

Do tính phức hợp phân cảnh tại môi trường hiện thực của người dùng, dự án ứng biến công nghệ Điều chuẩn Tinh chỉnh Học Sâu thông qua:

1. **Hiệu Ứng Dung Thông (Mosaic & Mixup):** Tăng sức đề kháng (Robustness configuration) của mạng với vật thể cắt lẹm / che khuất thông qua phép biến tính ma trận ảnh (Mixup = `0.1` | Mosaic = `1.0`).
2. **Dao Động Biến Ảnh Đa Chiều (Multi-scale Training bounds):** Gieo rắc biến thiên kích thước chập ngẫu nhiên quanh lõi chuẩn tĩnh `640` trong chu kỳ Epoch, tối đa hóa năng lực trực quan hệ thống.

> [!WARNING] Kiểm Thử Ma Trận Lỗi Xuyên Tâm (Error Matrix Defect Protocol)
> Tuyến luồng CLI tĩnh `python -m src.training.scripts.yolov8n predict-errors` là bắt buộc đối với mỗi vòng lập checkpoint epoch nhằm định danh các nhãn mờ sai cực (False Negatives / Overlapped labels) dựa vào ma trận so khớp Confidence Filtering.

## Kỷ Toán Lượng Tử Hóa Cạnh Biên (Quantization Aware Training - QAT)

Để nhúng thành công nhân rễ mô hình thô (FP32 precision constraints) lên dòng điện toán giới hạn (Edge limits computational threshold), bộ rẽ nhánh thực thi cấp bậc 1 yêu cầu ứng pháp QAT:

- Lắp đặt định quy Pytorch gốc `torch.quantization.prepare_qat()`.
- Chuyển giao đồ thị mô hình để giả lập rào cản tính toán nguyên khối `INT8 parameter drops` trong vòng 10-20 Epochs rặn trút rễ mạng giả học.
- Đúc kết biểu chuẩn so đo metric biểu hiệu năng đánh đổi `mAP@0.5` trước khi xác nhận lưu kho luồng đồ (Model graph export sequences).

> [!CAUTION] Cảnh Báo Tính Tương Thích Export Cục Vi (Export Dependencies Compatibility)
> Thao tác gọi luồng Lượng Tử Hóa không được can thiệp sâu thay thế cấu trúc rễ từ hệ sinh thái gói phần mềm Ultralytics mẹ (Dependency injection block bounds). Quá trình phân chia buộc bám lấy tài liệu chuẩn tĩnh `export()`.
