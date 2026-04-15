# Cấu Trúc Khối Mô Hình Đa Hợp (Models Ontology)

Là xương sống cốt lõi vận hành toàn bộ vòng đời suy luận (Inference Lifecycle) của Trợ lý Hỗ trợ Tiếp cận, hệ thống mô hình AIN501 được phân tách dựa trên các mục tiêu chuyên biệt để bám sát tính trồi (Emergent properties) trên thiết bị nhúng lề cạnh (Edge instances).

## Định Hình Phân Lớp Mô Hình (Models Classification)

Bảng phân loại dưới đây tổng hợp các hạt nhân tư duy học sâu trực thuộc dự án.

| Mã Nhận Diện Đồ Án (Model ID) | Nền Tảng Khởi Sinh (Base Architecture) | Tác Vụ Đặc Thù (Task Mapping) | Rào Cản Triển Khai (Edge Constraints) |
| --- | --- | --- | --- |
| **[YOLOv8-nano](yolov8n.md)** | ~3.2M params | Nhận diện đối tượng & chướng ngại vật (Object + hazard detection). | ~6MB Size (quantized) |
| **[MoViNet-A0](movinet.md)** | ~3M params | Nhận diện hành động (Action recognition). | ~5MB Size |
| **[MobileNetV3-Small](mobilenetv3.md)** | ~2.5M params | Bộ máy chiết xuất đặc trưng (Feature extraction). | ~4MB Size |
| **[all-MiniLM-L6](all_minilm.md)** | ~22M params | Tạo cấu trúc từ vựng vector (Embedding for RAG). | ~80MB Size |
| **[MediaPipe](mediapipe.md)** | ~1M params | Nhận diện mặt phẳng mặt & khung xương pose (Face mesh + pose). | ~3MB Size |
| **[PaddleOCR lite](paddleocr.md)** | ~4M params | Đọc văn bản tự nhiên & biển báo (Text / sign reading). | ~8MB Size |
| **[Scene classifier](scene_classifier.md)** | ~1M params | Bộ định tuyến đa nhiệm cảnh sắc (SLM router). | ~2MB Size |
| **[Piper TTS](piper_tts.md)** | Hệ tiếng viễn âm | Sinh giọng nói tương tác (Voice model vi/en). | ~50MB Size |

> [!NOTE] Cơ Chế Chuyển Tiếp Kháng Nước (Water-resistant Transfer Mechanism)
> Dự án áp dụng nguyên lý Lượng tử hóa trước/sau đào tạo (QAT/PTQ) và chiết giảm trọng số (Pruning parameters). Các điểm dữ liệu này vốn được nhương lại từ không gian đồ án C++ trên môi trường Edge.

## Giao Thức Đầu Vào / Ra (I/O Flow Conventions)

Hệ thống yêu cầu các ma trận đầu vào (Tensors input parameters) tuân thủ tính nghiêm minh học thuật để luân hồi được trên môi trường Multi-Agent.

- **Thị Giác Học (Vision Flow):** Đóng gói dưới hình hài Vector 3 chiều (C, H, W) hệ tọa độ màu RGB chuẩn tương hợp, dải nén điểm ảnh Normalize Mean/Std đặc tả. Kích thước biến quy dao động tiêu chuẩn ở mức điểm đồ `224x224` (MobileNet) tới `640x640` (YOLO).
- **Âm Học Biên Dịch (Audio Fallback):** Nhập liệu chữ viết cấu trúc trần (Raw String Text) với tốc độ chuỗi luân hồi tần số âm vĩ (Sample rate defaults) đặt ngưỡng 22kHz đơn nguyên (Mono).

> [!IMPORTANT] Tính Hợp Lệ Cấu Trúc Đồ Graph (Graph Validations)
> Mạng lưới (Backbones) bắt buộc phải vượt qua bài kiểm tra chép nhánh tĩnh OnnxRuntime (Dry-run conversion tracing) tại thời điểm khởi tạo nhằm ngừa gãy đổ luồng phân bổ (Deployment failures).

Tuần 6: benchmark thống nhất và registry triển khai tại [`models/registry.json`](../../models/registry.json); báo cáo `reports/week6/benchmark_report.md` và ghi chú [Week 6 training log](../training/week6_training_log.md).

Tiếp tục đi sâu phân tách từng Mô hình hệ cụ thể nằm trong mục lục chức năng đính kèm!
