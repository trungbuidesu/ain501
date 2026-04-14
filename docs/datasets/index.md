# Mục Lục Thư Viện Bộ Dữ Liệu (Datasets Index Directory)

Không gian này thiết lập bản đồ cấu trúc cấp vĩ mô đối với Hệ thống CSDL của AIN501.

## Động Thái Triển Khai Phương Thức (Deployment Paradigm)

> [!WARNING] Quy Tắc Lưu Trữ Ngoại Vi (External Tracking Bounds)
> - Cam kết quy trình Thử Nghiệm Mô Phỏng (Dry-run mode) luôn đi trước khi tác động lên cấu trúc lưu trữ cục bộ.
> - Cấu trúc dữ liệu phi kỹ thuật siêu lớn (Large raw artifacts) được cố định tại khối nội tại `data/`, `models/` và từ chối đồng bộ hóa phiên bản (No Git Tracking).

## Sơ Đồ Cây Bộ Dữ Liệu (Dataset Forest Topology)

| Nhóm Dự Năng (Group) | Điểm Giao Thức (Task) | Tham Số Cấu Hình Nguồn (Core Config) | Điểm Phát Sinh (Primary Output) | Xếp Hạng Ưu Tiên (Priority) |
| --- | --- | --- | --- | --- |
| Lõi (Core) | Phát hiện Đối Tượng | `object_detection_accessibility.yaml` | `data/processed/object_detection...` | Nền Tảng (Phases 0) |
| Lõi (Core) | Nhận diện Hành Động | `action_accessibility.yaml` | `data/processed/action_accessibility/...` | Nền Tảng (Phases 0) |
| Lõi (Core) | Phân Loại Cảnh Vật | `scene_accessibility.yaml` | `data/processed/scene_accessibility/...` | Nền Tảng (Phases 0) |
| Đới Mở (Extended) | Tách Chữ Cảnh OCR | `ocr_textocr.yaml` | `data/processed/ocr_textocr/...` | Phase 0 Mở rộng |
| Đới Mở (Extended) | Chú Phụ Đề Đoạn Băng | `video_captioning_msvd.yaml` | `data/processed/video_captioning...` | Chu Đai Kế Cận (Tier 2)|
| Đới Mở (Extended) | Đo Đạc Trình Chữ/Tiếng | `tts_piper_accessibility.yaml` | `data/tts_validation` | Xác thực Thuật Toán |
| Khối Dự Khuyết| Chiều Sâu Không Gian | `depth_midas_reserved.yaml` | `data/processed/depth/predictions` | Chờ Ấn Định (Reserved) |

## Chuỗi Liên Hệ Chuyên Đề Phân Quyền (Domain-Specific Hyperlinks)

- [Quy trình Phát Hiện Đối Tượng (Object Detection Architecture)](object_detection.md)
- [Quy trình Nhận Diện Hành Vi Học (Action Recognition Schema)](action_recognition.md)
- [Quy trình Phân Phối Định Hướng Môi Trường (Scene Classification Logic)](scene_classification.md)
- [Báo Khảo Cứu Dữ Liệu Ngoại Biên (Extended Modules Matrix)](extended.md)

## Chuỗi Hướng Dẫn Tác Trình (Canonical Methodology)

1. Giám định lộ trình tải xuống và đường dẫn không gian tĩnh (Local path bounds) qua `--dry-run`.
2. Đồng nhất khối thông dịch (Normalization parameter bounds).
3. Đánh giá thẩm sai phân dạng tĩnh (Validation mechanism) trước khi đưa cấu trúc vào vòng xoay Đào tạo (Training iterations).
4. Khai thông thẻ cấp cấp truy cập `--execute` sau khi vượt qua đánh giá an ninh quyền bảo mật và giới hạn ghi/đọc phần cứng (IO bottlenecks). 

## Tham Trạng Dữ Liệu Quan Chú Ý (Parametric Core References)

| Tiêu Điểm Tác Vụ | Hệ Quy Chiếu Cần Đặc Thù Quan Tâm (Core Variables) |
| --- | --- |
| Phát Hiện Đối Tượng | `bbox_format`, mảng cấu trúc `coco_subset/custom_only`, `canonical_names_path`. |
| Động Thái Nhận Diện | `frame_sequence_length`, `frame_stride`, điểm gieo mầm tổ hợp split `split_seed`. |
| Cảnh Vật Nhân Diện | `places_subset`, lớp phân giải ngoài quy chỉnh `custom_or_proxy`, `max_images_per_class`. |
| Khảm Chữ Vật Chất | `text_field`, `bbox_format`, lưới tổ hợp hình ảnh `polygon_field`. |
| Tạo Phụ Đề Chuyển Động | Khóa cấu trúc phân nhóm `normalized_mapping`. |
| Phản Biện Giọng Chữ | Chữ ký giọng nói fallback, tần số băng thông phát `expected_sample_rate_hz`. |
| Tính Thể Chiều Sâu | Giao diện thu phát `input_format / output_format`, tỷ lệ độ sâu `metric_distance`. |

## Hệ Sinh Thái Sổ Tay Phân Tích Hiện Hành (EDA Notebooks Mapping)

Hệ thống Sổ tay được chia làm hai mảng lớn: Khảo phân tĩnh thông số động (Dynamic Visuals) và Kiểm chuẩn tĩnh thuật toán (Sanity checks).

| Nhóm Tính Năng Trọng Yếu | Mỏ Neo Cú Pháp Notebook (Script Anchor) |
| --- | --- |
| Định Lượng Đối Tượng COCO | [0.3_coco_detection_explore.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.3_coco_detection_explore.ipynb) |
| Tần Suất Xếp Hành Động UCF | [0.4_action_ucf101_explore.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.4_action_ucf101_explore.ipynb) |
| Giới Hạn Mật Độ Môi Trường Places| [0.5_places_scene_subset.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.5_places_scene_subset.ipynb) |
| Ánh Xạ Phụ Đề MSVD | [0.6_msvd_captioning_explore.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.6_msvd_captioning_explore.ipynb) |
| Sàng Lọc Ký Tự Mờ TextOCR | [0.7_textocr_explore.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.7_textocr_explore.ipynb) |
| Gắn Nhãn Phát Thanh Piper TTS | [0.8_tts_piper_validation.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.8_tts_piper_validation.ipynb) |
| Cây Giữ Chỗ MiDaS (Reserved) | [0.9_depth_midas_reserved.ipynb](file:///d:/workspace/GitHub/ain501/notebooks/0.9_depth_midas_reserved.ipynb) |

## Thiết Chế Chất Lượng Tài Nguyên (Data Ethics Principles)

1. Tối thượng hóa quyền sử dụng nguồn đã được Đồng bộ/Chuẩn hóa (Normalized arrays) so với nguồn thô mới giải nén (Raw extraction).
2. Dữ liệu Tự Ghi Hình (Custom capture) buộc phải vượt qua hàng rào Kiểm Duyệt Dữ Liệu Cá nhân (Privacy Sanitization Workflow) trước lúc Đánh Nhãn.
3. Chặn đứng chu trình đánh giá khi xuất hiện lỗi Biệt lệ Tích Giao (Validation leakage) tại kho tàng đánh giá hiện thời.
