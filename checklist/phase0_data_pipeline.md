# Checklist Phase 0 - Data Pipeline Accessibility Assistant

File này dùng để theo dõi tiến độ Phase 0. Quy ước:

- `[x]` đã implement trong repo hoặc đã hoàn thành thực tế.
- `[ ]` chưa hoàn thành.
- `Trạng thái`: ghi chú ngắn để biết đang ở mức scaffold, dry-run hay đã chạy dữ liệu thật.

## Object Detection Dataset

- [ ] Download COCO 2017 train + val (~20GB)
  - Trạng thái: Chưa download dữ liệu thật. Đã có config nguồn và `download-plan` trong `configs/datasets/object_detection_accessibility.yaml`.
- [x] Hiểu COCO annotation format (JSON, bbox, categories)
  - Trạng thái: Đã implement parser/converter COCO bbox `[x, y, width, height]` sang YOLO trong `training/scripts/data_utils.py`, có test fixture.
- [ ] Viết data exploration notebook: phân tích distribution classes, bbox sizes
  - Trạng thái: Mới có notebook starter `notebooks/0.3_coco_detection_explore.ipynb`. Chưa có code phân tích distribution/bbox đầy đủ và chưa chạy trên COCO local.

### Optional - Custom Domain Data

- [ ] Record screen 30-60 phút với các scenario khác nhau
  - Trạng thái: Chưa record thật. Đã có script dry-run `training/scripts/capture_frames.py`.
- [ ] Annotate bằng CVAT hoặc Roboflow (200-500 images)
  - Trạng thái: Chưa annotate thật. Đã có workflow trong `docs/data_pipeline.md`.
- [ ] Convert annotation sang YOLO format
  - Trạng thái: Đã có COCO -> YOLO converter và YOLO validator trong `training/scripts/data_utils.py`. Chưa có converter CVAT/Roboflow raw export -> YOLO hoàn chỉnh và chưa chạy với export thật.
- [x] Merge với subset COCO nếu cần
  - Trạng thái: Đã implement `merge-yolo --dry-run` để preview remap trước khi ghi file. Chưa merge dataset thật.

## Action Recognition Dataset

- [ ] Download UCF-101 hoặc HMDB-51
  - Trạng thái: Chưa download dữ liệu thật. Plan chọn HMDB-51 làm bootstrap trong `configs/datasets/action_accessibility.yaml`.
- [ ] Viết data loader: video -> frame sequences
  - Trạng thái: Chưa implement data loader hoàn chỉnh. Mới có split helper và config sampling.
- [x] Phân tích: chọn 20-30 action classes phù hợp use case
  - Trạng thái: Đã chọn class public bootstrap và custom-required trong `configs/datasets/action_accessibility.yaml`. `approaching` được ghi rõ là extension Phase 9, không block Phase 0.
- [x] Train/val/test split
  - Trạng thái: Đã implement helper `stratified_split` và `group_aware_split` trong `training/scripts/data_utils.py`, có test.

## Scene Classification Dataset

- [ ] Download Places365 subset mini (~6GB)
  - Trạng thái: Chưa download dữ liệu thật. Đã có config trong `configs/datasets/scene_accessibility.yaml`.
- [x] Chọn 10-20 scene categories phù hợp
  - Trạng thái: Đã chọn subset scene trong `configs/datasets/scene_accessibility.yaml`, gồm cả note cho `sidewalk` và `bus_stop` là custom/proxy.
- [ ] Tạo balanced subset cho training router
  - Trạng thái: Chưa tạo subset thật. Đã có config balancing và notebook starter `notebooks/0.5_places_scene_subset.ipynb`.
- [ ] Optional: tự capture + label thêm 200-500 screenshots
  - Trạng thái: Chưa capture/label thật. Đã có privacy checklist và capture script dry-run.

## Video Captioning Dataset (Tier 2)

- [ ] Download MSR-VTT hoặc MSVD
  - Trạng thái: Chưa download dữ liệu thật. Đã có config `configs/datasets/video_captioning_msr_vtt.yaml`.
- [ ] Explore: xem sample videos + captions
  - Trạng thái: Mới có notebook starter `notebooks/0.6_msr_vtt_captioning_explore.ipynb`. Chưa xem sample thật vì chưa có dataset local.
- [x] Hiểu format: `video_id -> multiple captions`
  - Trạng thái: Đã implement parser normalize MSR-VTT thành `video_id -> captions[]`, có test fixture.

## OCR Dataset

- [ ] Download ICDAR 2015 hoặc TextOCR
  - Trạng thái: Chưa download dữ liệu thật. Đã có config `configs/datasets/ocr_icdar2015.yaml`.
- [ ] Explore: các loại text (scene text, document, handwriting)
  - Trạng thái: Mới có notebook starter `notebooks/0.7_icdar_ocr_explore.ipynb` và parser ICDAR fixture. Chưa explore dataset thật.

## Data Pipeline Utilities (`training/scripts/data_utils.py`)

### Augmentation Pipeline

- [ ] Random crop, flip, color jitter, brightness
  - Trạng thái: Chưa đủ. Hiện mới có `augment-preview` đơn giản với resize + horizontal flip.
- [ ] Resize + normalize chuẩn cho từng model
  - Trạng thái: Chưa đủ. Cần bổ sung model-specific transforms cho detection/classification/action/OCR.

### Split And Validation

- [x] Viết train/val splitter
  - Trạng thái: Đã có `stratified_split`, `group_aware_split`, CLI `split`, và tests.
- [x] Viết data validation script (check corrupt images, missing labels)
  - Trạng thái: Đã có YOLO validation check corrupt image, missing label/image, class id và bbox range. Chưa validate video corrupt đầy đủ.

## Bổ Sung Đã Thêm Ngoài Checklist Gốc

- [x] Config Piper TTS Vietnamese + English
  - Trạng thái: Đã có `configs/datasets/tts_piper_accessibility.yaml`.
- [x] Validate TTS dry-run
  - Trạng thái: Đã có CLI `validate-tts --dry-run` và WAV validation helper.
- [x] Reserve config slot cho MiDaS depth estimation
  - Trạng thái: Đã có `configs/datasets/depth_midas_reserved.yaml` và notebook placeholder `notebooks/0.9_depth_midas_reserved.ipynb`.
- [x] Privacy checklist cho outdoor capture
  - Trạng thái: Đã có trong `docs/data_pipeline.md`.

## Tổng Kết Hiện Tại

- Hoàn thành scaffolding/config/docs/notebook starter/utility dry-run/test fixtures.
- Chưa hoàn thành download dataset thật, capture/annotate thật, balanced subset thật, converter CVAT/Roboflow raw export -> YOLO, và analysis notebook chạy trên dữ liệu thật.
- Hai phần code nên ưu tiên tiếp theo: video frame-sequence loader và augmentation pipeline đầy đủ.
