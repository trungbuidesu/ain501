# Phương Tố Thực Nghiệm Khoa Học (Experiments Documentation)

Trang tài liệu này quy hoạch các tiêu chuẩn khắc nghiệt nhất cho điểm điều phối theo dõi lịch sử và cấu hình bảo tồn quản trị thông tin lưu log (Artifact management parameters) cho tất thảy các cuộc huấn luyện khảo cứu nội bộ.

## Quy Tắc Lập Danh Xưng (Tagging and Naming Convention)

Mọi chạy thực nghiệm đồ tạo (Runs) trên TensorBoard hay Weights & Biases đều buộc bám sát ma trận cấu hình nhãn dán:

`[Tác vụ]_[Mô hình]_[Loại Dataset]_[Thông số siêu vi]_[Phiên bản]`

**Ví dụ:** `det_yolov8n_cocosubset_lr001_v1`
- `det`: Tác vụ nhận diện (Detection).
- `yolov8n`: Mạng trọng khối thuật rễ.
- `cocosubset`: Nguồn hạt giống (Dataset root alias).
- `lr001_v1`: Hệ thông số độ dốc (Learning rate) cùng cấu mốc version thực thể.

## Nguyên Tắc Khóa Cố Định Hạt Giống Đào Tạo (Seed Locking Protocol)

Để bảo đảm tính truy hồi tương xứng tuyệt đối (Absolute Reproducibility), nhà nghiên cứu buộc phải cưỡng chế ép cứng thông số hạt ngẫu nhiên (Fixed random seeds constraint) trước mọi thuật chia xuất thông mạch:

- `seed: 42` hoặc `1337` được cài cắm khắp mọi khâu chia tỉ lệ mảng tập tin đồ hình (Train/Val Split Array Logic).
- Bảo an hạt giống (Seed hooks array parameter limits logical structure map variables functions constraints parameters format routing format list text function mapped bindings logic module function variables protocol limits format mapping) củng cố lại sự đan chéo đồng bộ qua mạc Torch/Numpy/Random gốc lệnh rễ biến hóa đồ cấu biến.

## Biểu Đồ Quản Trị Di Sản Lập (Artifact Checkpoint Lifecycles)

> [!WARNING] Cảnh Báo Thu Dọn Tàn Dư Đồ Đệ Phương Phẳng (Artifact Log Dumps Map Warning Parameters Data Arrays Rule limits Function Data limits Layout variable logic Rules Bounds Data Parameter function Format variable logical bounds)
> Máy chủ nội bộ sẽ nứt tràn dữ liệu cực nhanh nếu không cài đặt định quy vòng xóa tự rễ (Garbage array mapping function parameters limit boundaries rules bounds matrix data config layout string matrix parsing array variables bounds bounds string routing variables layout strings limits parameters logic list).

Mọi checkoints biến (`.pt` / `.onnx`) qua vòng Epoch tốt bụng nhất (`best.pt`) được xuất lên `runs/models_archived/` với đuôi chỉ mục Version ngắt ngang đồ mạng. Chỉ những siêu log giá trị biên độ lớn mới được chuyển hóa chéo về định dạng nhị phân Tensorboard (Tensorboard scalar events map logic bounds formatting node limits variables structure matrix function variables parameters bounds).
