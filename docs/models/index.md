# Cấu Trúc Khối Mô Hình Đa Hợp (Models Ontology)

Là xương sống cốt lõi vận hành toàn bộ vòng đời suy luận (Inference Lifecycle) của Trợ lý Hỗ trợ Tiếp cận, hệ thống mô hình AIN501 được phân tách dựa trên các mục tiêu chuyên biệt để bám sát tính trồi (Emergent properties) trên thiết bị nhúng lề cạnh (Edge instances).

## Định Hình Phân Lớp Mô Hình (Models Classification)

Bảng phân loại dưới đây tổng hợp các hạt nhân tư duy học sâu trực thuộc dự án.

| Mã Nhận Diện Đồ Án (Model ID) | Nền Tảng Khởi Sinh (Base Architecture) | Tác Vụ Đặc Thù (Task Mapping) | Rào Cản Triển Khai (Edge Constraints) |
| --- | --- | --- | --- |
| **[Biên Dịch Vật Thể (YOLOv8n)](yolov8n.md)** | `ultralytics/yolov8n` | Đóng khung giới hạn (Object Detection) đối với các vật tư chướng ngại công cộng. | Kích thước < 7MB, FP16/INT8, Tần số nội tại CPU/NPU. |
| **[Xác Thực Không Gian (MobileNetV3)](mobilenetv3.md)** | `torchvision/mobilenet_v3_small` | Tách cụm và định tuyến nhãn quan (Scene/Action Classification) phục vụ cảnh báo bối cảnh. | Loại trừ MLP Head, Tối thiểu hóa kích thước Vector Embedding (576D). |
| **[Kiến Trúc Đọc Âm Mộc (TTS Piper)](piper_tts.md)** | `rhasspy/piper/vits` | Kích hoạt đa luồng thông thư đàm thoại cảnh báo người dùng khiếm khuyết. | Lưu lượng âm tuyến cục bộ (On-device audio chunking). |

> [!NOTE] Cơ Chế Chuyển Tiếp Kháng Nước (Water-resistant Transfer Mechanism)
> Dự án áp dụng nguyên lý Lượng tử hóa trước/sau đào tạo (QAT/PTQ) và chiết giảm trọng số (Pruning parameters). Các điểm dữ liệu này vốn được nhương lại từ không gian đồ án C++ trên môi trường Edge.

## Giao Thức Đầu Vào / Ra (I/O Flow Conventions)

Hệ thống yêu cầu các ma trận đầu vào (Tensors input parameters) tuân thủ tính nghiêm minh học thuật để luân hồi được trên môi trường Multi-Agent.

- **Thị Giác Học (Vision Flow):** Đóng gói dưới hình hài Vector 3 chiều (C, H, W) hệ tọa độ màu RGB chuẩn tương hợp, dải nén điểm ảnh Normalize Mean/Std đặc tả. Kích thước biến quy dao động tiêu chuẩn ở mức điểm đồ `224x224` (MobileNet) tới `640x640` (YOLO).
- **Âm Học Biên Dịch (Audio Fallback):** Nhập liệu chữ viết cấu trúc trần (Raw String Text) với tốc độ chuỗi luân hồi tần số âm vĩ (Sample rate defaults) đặt ngưỡng 22kHz đơn nguyên (Mono).

> [!IMPORTANT] Tính Hợp Lệ Cấu Trúc Đồ Graph (Graph Validations)
> Mạng lưới (Backbones) bắt buộc phải vượt qua bài kiểm tra chép nhánh tĩnh OnnxRuntime (Dry-run conversion tracing) tại thời điểm khởi tạo nhằm ngừa gãy đổ luồng phân bổ (Deployment failures).

Tiếp tục đi sâu phân tách từng Mô hình hệ cụ thể nằm trong mục lục chức năng đính kèm!
