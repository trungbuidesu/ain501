# Trung Tâm Lưu Trữ Khảo Cứu AIN501 (AIN501 Documentation Hub)

Chào mừng độc giả đến với Cổng tri thức lõi của dự án AIN501 (được khởi tạo dựa trên kiến trúc MkDocs Material).

## Giao Trình Mục Tiêu (Objective Paradigm)

- Khởi tạo kiến trúc đa điểm quy tụ mọi Tài Liệu Kỹ Thuật (Technical Documentation).
- Phân rã cấu trúc theo Hệ Sinh Thái Chức Năng (Functional Ecosystem): Tổng quan kiến trúc, Cơ chế xử lý nguồn liệu, và Tuyến tiến trình kỹ thuật.
- Đảm bảo tính kế thừa liên tục từ Tuyên bố Cấu hình Tạm thời sang Ấn bản Khoa học Chính thức.

Trang tài liệu phân cấp dựa trên Khối Kiến Trúc Trọng Điểm:
- Khu vực `Tài Liệu Cốt Lõi`: Chứa Bảng Khái Quát (Overview), Phương Tố Thí Nghiệm (Experiments), Liên kết Mô Hình (Models), và Trung Tâm Dữ Liệu (Datasets).
- Khu vực `Tài Liệu Khai Triển`: Hội tụ các phân đoạn tiến trình thực thi (Pipeline configurations) chi tiết hiện hành.

## Trạng Thái Hiện Tại (Project Status - Phase 0)

Dự án đã hoàn thành các cột mốc quan trọng sau:
- **Action Recognition**: Huấn luyện thành công MoViNet-A0 trên Intel XPU, đạt độ chính xác ~62.6%. Đã hoàn tất nén INT8 (4.4MB).
- **Object Detection**: Pilot training YOLOv8n đã hoàn thành, thiết lập baseline cho việc nhận diện vật thể hỗ trợ tiếp cận.
- **Data Pipeline**: Chuyển đổi sang kiến trúc Frame-based giúp tối ưu hóa 90% tải CPU trên luồng video.

> [!NOTE] Cổng Siêu Dữ Liệu Lõi
> Vui lòng truy cập **[Trung Tâm Chỉ Mục Bộ Dữ Liệu (Datasets Index)](datasets/index.md)** để nắm rõ toàn cảnh Phương pháp Xử lý Cơ Sở Dữ Liệu Phase 0.
