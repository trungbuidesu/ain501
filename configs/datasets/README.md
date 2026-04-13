# Cấu Hình Bộ Dữ Liệu (Dataset Configurations)

Thư mục này chứa toàn bộ các tệp cấu hình (YAML) đóng vai trò là xương sống định hình nguồn dữ liệu cho Phase 0 của trợ lý hỗ trợ tiếp cận (Accessibility Assistant). Các file cấu hình sẽ được đọc tự động bởi hệ thống pipeline `src/training/scripts/data_utils.py`. Hệ thống hoạt động theo nguyên tắc **Dry-run by default** (không lưu/tải dữ liệu lớn nếu không có cờ `--execute` xác nhận).

_Lưu ý: Dữ liệu thật (archive file, video, ảnh raw, label) không được đẩy lên Git mà nằm trong thư mục `data/` và `models/` trên máy local._

---

## Danh Sách và Đặc Tính Các Bộ Dữ Liệu (Datasets)

Dự án sử dụng cơ chế kết hợp giữa một Bootstrap Public Dataset lớn và bổ sung thêm Custom Subset cho từng tác vụ riêng biệt. Dưới đây là phân tích chi tiết dựa trên nội dung config thực tế:

### 1. Object Detection (Phát hiện đối tượng rào cản)
Tệp config: `object_detection_accessibility.yaml` (Phase 0.3)
- **Mục đích:** Tác vụ lõi giúp Agent kiểm soát vùng an toàn, "nhìn" thấy các vật cản, phương tiện, và đối tượng cần chú ý xung quanh lộ trình của người khiếm thị.
- **Nguồn:** `COCO 2017` (hỗ trợ kéo tự động file zip từ trang chủ, kích thước gốc ~20GB cho cả annotations, train2017, val2017).
- **Đặc tính/Bộ nhớ:**
  - Định dạng bbox COCO chuẩn: `coco_xywh_pixels` (sẽ được pipeline convert tự động sang `ultralytics_yolo`).
  - Gồm 16 Subset Class trọng yếu: person, dog, car, train, traffic light, stop sign, stairs...
  - Hỗ trợ custom workflow yêu cầu capture bổ sung 200-500 ảnh thực tế cho các nhóm bị thiếu ở COCO (elevator, crosswalk_signal, sidewalk_obstacle) để gán nhãn qua Roboflow/CVAT.

### 2. Action Recognition (Nhận diện hành vi)
Tệp config: `action_accessibility.yaml` (Phase 0.4)
- **Mục đích:** Đặc trị nhận diện đối tượng đang thực hiện hành động gì dạng video chuỗi (chu kỳ liên tiếp).
- **Nguồn:** `UCF-101` (video clips) với folder split chuẩn của ucfTrainTestlist.
- **Đặc tính/Bộ nhớ:**
  - Sử dụng cơ chế Sampling chặt chẽ: `frame_sequence_length: 16` (độ dài chuỗi 16 hình) và `frame_stride: 2` (interval lấy khung hình nhảy bước 2) cùng base seed `501`.
  - Khởi tạo với dãy 20 class Bootstrap có sẵn (v.d. WalkingWithDog, Biking...) và đặt trước config list cho 8 nhóm quan trọng sắp capture (`person_approaching`, `stairs_up`...).

### 3. Scene Classification (Nhận diện không gian/ngữ cảnh)
Tệp config: `scene_accessibility.yaml` (Phase 0.5)
- **Mục đích:** Trả lời trực quan câu hỏi "Chúng ta đang ở đâu?" - để Agent cảnh báo hay khuyên nhủ dựa trên việc đang ở bến xe buýt, ở nhà ga tàu điện hay đang trên công viên.
- **Nguồn:** `Places365` (ưu tiên bản crop `places365_standard_256` kích thước 256px nhẹ và dễ modelizing hơn bản High-res).
- **Đặc tính/Bộ nhớ:**
  - Filter một list 16 bối cảnh sát sườn nhất (crosswalk, living_room, parking_lot, street...).
  - Quản lý kích thước dữ liệu nghiêm khắc bằng hệ thống Balancing: chặn mức scale tối đa `max_images_per_class: 1000` ảnh/nhãn tránh làm file manifest đầu ra quá tải.

### 4. Tiện ích và Dự Phòng Tương Lai (Phase 0.6 -> 0.9)

- **Video Captioning (`video_captioning_msr_vtt.yaml`) - Phase 0.6:**
  - Mục đích: Bình luận và tóm tắt theo thời gian thực mô tả hoạt cảnh. Dữ liệu: `MSR-VTT`. Tier ưu tiên thấp (Tier 2 deferred), cơ chế tự động mapping normalized format `video_id -> captions[]`.

- **OCR Scene Text (`ocr_icdar2015.yaml`) - Phase 0.7:**
  - Mục đích: Đọc nội dung tự nhiên không dùng nét ngay ngắn ngoài thực tiễn, số nhà, hoặc label thang máy. Dữ liệu trích ly: `ICDAR 2015 Challenge 4`. Cấu hình định dạng box bounding 4 góc chéo (`quadrilateral_points: 4`) và chỉ định việc bỏ qua text loang lổ / vỡ hình theo luật (`###`).

- **Validation Text-to-Speech (`tts_piper_accessibility.yaml`) - Phase 0.8:**
  - Mục đích: Validate chất âm báo bằng lời module phản hồi cho người khiếm thị. Config nhúng đường dẫn các model `.onnx` tiếng Việt (`vi_VN-vais1000-medium`) và tiếng Anh với sample rate `22050Hz`. Hỗ trợ script fallback model.

- **Depth Estimation (`depth_midas_reserved.yaml`) - Phase 0.9:**
  - Mục đích: Ước lượng chiều sâu qua một ảnh duy nhất, hỗ trợ nhận định cự ly vật lý. Reserved file. Cấu trúc yêu cầu `input_format: RGB image` ra `output_format: relative_depth_map` (Trạng thái đánh dấu: không cản trở block Phase 0 core).

---

## Lệnh Pipeline Workflow (Cheatsheet)

Module `src.training.scripts.data_utils` là trung tâm xử lý dữ liệu. Một số lệnh cơ bản bạn sẽ cần:

```bash
# 1. Kiểm tra plan download / tải dataset tự động dựa vào file config tải archive
python -m src.training.scripts.data_utils download-plan

# 2. Verify check validation tình hình folder root chứa các bản unpack (giống như "git status" của dataset vậy)
python -m src.training.scripts.data_utils verify-dataset-paths --skip-downloads

# 3. Trích xuất text Class chung cho COCO và setup COCO Object Detection
python -m src.training.scripts.data_utils write-classes --execute
python -m src.training.scripts.data_utils convert --coco-json data/external/coco2017/annotations/instances_val2017.json --image-root data/external/coco2017/val2017 --output-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt --split val --execute

# 4. Generate Balanced CSV của Scene Places365
python -m src.training.scripts.data_utils build-scene-subset --file-list data/external/places365/places365_val.txt --execute

# 5. Check Output cuối cho mọi dữ liệu đã build
python -m src.training.scripts.data_utils validate --dataset-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt
python -m src.training.scripts.data_utils validate-tts --dry-run
```
