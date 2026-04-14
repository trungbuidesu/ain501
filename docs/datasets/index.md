# Bộ Dữ Liệu

Trang này là điểm vào cho tài liệu bộ dữ liệu của AIN501.

## Phạm Vi

- Ưu tiên workflow dry-run trước khi ghi dữ liệu thực tế.
- Tách riêng tài liệu core datasets và nhóm mở rộng/dự phòng.
- Giữ artifact lớn ở local (`data/`, `models/`) và không đưa lên Git.

## Bản Đồ Bộ Dữ Liệu

| Nhóm | Tác vụ | Config chính | Output chính | Mức ưu tiên |
| --- | --- | --- | --- | --- |
| Core | Object Detection (Phát hiện đối tượng) | `configs/datasets/object_detection_accessibility.yaml` | `data/processed/object_detection_accessibility` | Phase 0 bắt buộc |
| Core | Action Recognition (Nhận diện hành động) | `configs/datasets/action_accessibility.yaml` | `data/processed/action_accessibility/manifest.csv` | Phase 0 bắt buộc |
| Core | Scene Classification (Phân loại bối cảnh) | `configs/datasets/scene_accessibility.yaml` | `data/processed/scene_accessibility/manifest.csv` | Phase 0 bắt buộc |
| Mở rộng | OCR Scene Text (Nhận diện chữ trong cảnh) | `configs/datasets/ocr_textocr.yaml` | `data/processed/ocr_textocr/annotations.jsonl` | Phase 0 mở rộng |
| Mở rộng | Video Captioning (Chú thích video) | `configs/datasets/video_captioning_msvd.yaml` | `data/processed/video_captioning_msvd/captions.json` | Tier 2 |
| Mở rộng | TTS Validation (Kiểm tra TTS) | `configs/datasets/tts_piper_accessibility.yaml` | `data/tts_validation` | Validation pipeline |
| Dự phòng | Depth Estimation (Ước lượng độ sâu) | `configs/datasets/depth_midas_reserved.yaml` | `data/processed/depth/predictions` | Reserved |

## Nhóm Tài Liệu Dataset

- [Object Detection (Phát hiện đối tượng)](object_detection.md)
- [Action Recognition (Nhận diện hành động)](action_recognition.md)
- [Scene Classification (Phân loại bối cảnh)](scene_classification.md)
- [Mở rộng (OCR, Chú thích video, TTS, Độ sâu)](extended.md)

## Quy Trình Chuẩn Nên Theo

1. Kiểm tra kế hoạch tải và path local bằng dry-run.
2. Chuẩn hóa dữ liệu về format mà từng task yêu cầu.
3. Chạy validation cho output trước khi train.
4. Chỉ dùng `--execute` khi đã kiểm tra dung lượng lưu trữ và điều kiện riêng tư.

```bash
python -m src.training.scripts.data_utils download-plan
python -m src.training.scripts.data_utils verify-dataset-paths --tasks object_detection action_recognition scene_classification --skip-downloads
```

## Tham Số Cốt Lõi Theo Từng Nhóm

| Tác vụ | Tham số cần chú ý |
| --- | --- |
| Object Detection | `bbox_format`, danh sách `coco_subset/custom_only`, `canonical_names_path`, `export_format` |
| Action Recognition | `frame_sequence_length`, `frame_stride`, `split_seed`, class bootstrap/custom |
| Scene Classification | `places_subset`, `custom_or_proxy`, `max_images_per_class`, `split_seed` |
| OCR | `text_field`, `bbox_format`, `polygon_field`, điều kiện bỏ qua annotation |
| Captioning | `normalized_mapping`, trường accepted_fields |
| TTS | voice id mặc định/fallback, `expected_sample_rate_hz`, validation checks |
| Depth | `input_format`, `output_format`, `metric_distance` |

## Quy Ước Thư Mục Dùng Chung

- `data/downloads/`: chứa archive tải từ nguồn chính thức.
- `data/external/`: dữ liệu bên thứ ba đã giải nén.
- `data/custom/raw_frames_private/`: frame thô có thể chứa thông tin nhạy cảm.
- `data/custom/sanitized_frames/`: frame đã qua review riêng tư.
- `data/custom/annotation_exports/`: export từ CVAT hoặc Roboflow.
- `data/processed/`: output đã convert/validate để train.
- `models/voices/piper/`: artifact voice Piper local.

## Nguyên Tắc Chất Lượng Dữ Liệu

- Ưu tiên dữ liệu đã normalize thay vì dùng trực tiếp raw export.
- Với dữ liệu custom, phải có bước review riêng tư trước annotation.
- Không dùng artifact chưa validate để benchmark hoặc train dài.
- Mọi thay đổi logic xử lý dữ liệu cần cập nhật lại tài liệu tương ứng trong `docs/datasets/`.

## Tài Liệu Liên Quan

- Vận hành pipeline: `docs/data_pipeline.md`
- Tham chiếu config dataset: `configs/datasets/README.md`
