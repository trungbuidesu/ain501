# Bộ Dữ Liệu

Trang này là điểm vào cho tài liệu bộ dữ liệu của AIN501.

## Phạm Vi

- Ưu tiên workflow dry-run trước khi ghi dữ liệu thực tế.
- Tách riêng tài liệu core datasets và nhóm mở rộng/dự phòng.
- Giữ artifact lớn ở local (`data/`, `models/`) và không đưa lên Git.

## Nhóm Tài Liệu Dataset

- [Object Detection (Phát hiện đối tượng)](object_detection.md)
- [Action Recognition (Nhận diện hành động)](action_recognition.md)
- [Scene Classification (Phân loại bối cảnh)](scene_classification.md)
- [Mở rộng (OCR, Chú thích video, TTS, Độ sâu)](extended.md)

## Quy Ước Thư Mục Dùng Chung

- `data/downloads/`: chứa archive tải từ nguồn chính thức.
- `data/external/`: dữ liệu bên thứ ba đã giải nén.
- `data/custom/raw_frames_private/`: frame thô có thể chứa thông tin nhạy cảm.
- `data/custom/sanitized_frames/`: frame đã qua review riêng tư.
- `data/custom/annotation_exports/`: export từ CVAT hoặc Roboflow.
- `data/processed/`: output đã convert/validate để train.
- `models/voices/piper/`: artifact voice Piper local.

## Tài Liệu Liên Quan

- Vận hành pipeline: `docs/data_pipeline.md`
- Tham chiếu config dataset: `configs/datasets/README.md`
