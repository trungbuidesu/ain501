# Checklist Phase 0 - Data Pipeline Accessibility Assistant

File này dùng để theo dõi tiến độ Phase 0. Quy ước:

- `[x]` đã implement trong repo hoặc đã hoàn thành thực tế.
- `[ ]` chưa hoàn thành.
- `Trạng thái`: ghi chú ngắn để biết đang ở mức scaffold, dry-run hay đã chạy dữ liệu thật.

## Object Detection Dataset

- [x] Download COCO 2017 train + val (~20GB)
  - Trạng thái: Đã tải/extract `train2017.zip`, `val2017.zip` và `annotations_trainval2017.zip` vào `data/external/coco2017`. Hiện có 118287 ảnh train, 5000 ảnh val; đã ghi `classes.txt`, convert COCO val -> YOLO với 3445 ảnh selected, tạo merged YOLO root và validate YOLO `checked_files=6890 valid=True`.
- [x] Hiểu COCO annotation format (JSON, bbox, categories)
  - Trạng thái: Đã implement parser/converter COCO bbox `[x, y, width, height]` sang YOLO trong `src/training/scripts/data_utils.py`, có test fixture.
- [x] Viết data exploration notebook: phân tích distribution classes, bbox sizes
  - Trạng thái: `notebooks/0.3_coco_detection_explore.ipynb` đã có output text/table nhỏ cho COCO train+val thật: image counts, 16-class distribution, bbox width/height/area summary, và YOLO conversion counts. Dataset images/labels vẫn nằm trong `data/` ignored.

### Optional - Custom Domain Data

- [ ] Record screen 30-60 phút với các scenario khác nhau
  - Trạng thái: Chưa record thật. Đã có script dry-run `src/training/scripts/capture_frames.py`.
- [ ] Annotate bằng CVAT hoặc Roboflow (200-500 images)
  - Trạng thái: Chưa annotate thật. Đã có workflow trong `docs/data_pipeline.md`.
- [ ] Convert annotation sang YOLO format
  - Trạng thái: Đã có COCO -> YOLO converter, YOLO validator, và CLI `normalize-custom-yolo` cho export YOLO dạng Roboflow/CVAT/Ultralytics với remap class theo `classes.txt`/`obj.names`/`data.yaml`, copy về layout canonical `images/<split>` + `labels/<split>`, ghi `classes.txt`, và validate sau khi `--execute`. Còn mở vì chưa có export custom thật sau privacy review + annotation để ingest.
- [x] Merge với subset COCO nếu cần
  - Trạng thái: Đã implement `merge-yolo`; hiện đã merge COCO YOLO subset vào `data/processed/object_detection_accessibility_merged` với 3445 ảnh và 3445 label, validate `checked_files=6890 valid=True`. Chưa có custom export thật để merge thêm.

## Action Recognition Dataset

- [x] Download UCF-101
  - Trạng thái: Đã tải/extract UCF-101 vào `data/external/ucf101`; hiện có 13320 video dưới `UCF-101` và official split files dưới `ucfTrainTestlist`.
- [x] Viết data loader: video -> frame sequences
  - Trạng thái: Đã implement action manifest builder cho class-folder videos, official split parser cho UCF-101, frame-sequence sampler và video manifest validation trong `src/training/scripts/data_utils.py`, có test fixture. Đã chạy trên UCF-101 thật và tạo `data/processed/action_accessibility/manifest.csv` với 2449 rows.
- [x] Phân tích: chọn 20-30 action classes phù hợp use case
  - Trạng thái: Đã chọn class public bootstrap và custom-required trong `configs/datasets/action_accessibility.yaml`. `approaching` được ghi rõ là extension Phase 9, không block Phase 0.
- [x] Train/val/test split
  - Trạng thái: Đã implement helper `stratified_split` và `group_aware_split` trong `src/training/scripts/data_utils.py`, có test.

## Scene Classification Dataset

- [x] Download Places365 subset mini (~6GB)
  - Trạng thái: Đã tải/extract official Places365 devkit/filelist và `val_256` 256px vào `data/external/places365`. Đây là bootstrap subset 36,500 ảnh val; chưa tải full train 256px.
- [x] Chọn 10-20 scene categories phù hợp
  - Trạng thái: Đã chọn subset scene trong `configs/datasets/scene_accessibility.yaml`, gồm cả note cho `sidewalk` và `bus_stop` là custom/proxy.
- [x] Tạo balanced subset cho training router
  - Trạng thái: Đã chạy `build-scene-subset --file-list data/external/places365/places365_val.txt --execute`, tạo `data/processed/scene_accessibility/manifest.csv` với 1600 rows = 16 class x 100 ảnh. Chưa copy ảnh sang processed để tránh nhân đôi dữ liệu.
- [x] Optional: tự capture + label thêm 200-500 screenshots
  - Trạng thái: Không còn track như một mục mở riêng của scene classification. Nếu cần data custom thật, dùng cùng human-gated workflow ở phần Custom Domain Data: record/capture, privacy review, annotate, rồi ingest bằng `normalize-custom-yolo`.

## Video Captioning Dataset (Tier 2)

- [x] Download MSVD
  - Trạng thái: Đã tải MSVD từ Hugging Face dataset mirror `friedrichor/MSVD` vào `data/downloads/msvd/hf_snapshot`, extract 1970 video vào `data/external/msvd/videos`, gộp annotation vào `data/external/msvd/annotations/msvd_captions.json` và normalize `data/processed/video_captioning_msvd/captions.json` với 1970 video / 80827 captions.
- [x] Explore: xem sample videos + captions
  - Trạng thái: `notebooks/0.6_msvd_captioning_explore.ipynb` đã có output text/table nhỏ cho MSVD: 1970 video, 80827 captions, captions-per-video stats, caption length stats, và 5 sample `video_id -> captions[]` rows. Không commit video.
- [x] Hiểu format: `video_id -> multiple captions`
  - Trạng thái: Đã implement parser normalize MSVD thành `video_id -> captions[]`, có test fixture.

## OCR Dataset

- [x] Download TextOCR
  - Trạng thái: Đã tải/extract TextOCR vào `data/external/textocr`; hiện có 25119 ảnh trong `train_val_images`, `TextOCR_0.1_train.json`, `TextOCR_0.1_val.json`, và normalized output `data/processed/ocr_textocr/annotations.jsonl` với 1202339 text boxes.
- [x] Explore: các loại text (scene text, document, handwriting)
  - Trạng thái: `notebooks/0.7_textocr_explore.ipynb` đã có output text/table nhỏ cho TextOCR: 25119 image count, train/val text-box counts, text length stats, ignored/illegible rate, bbox size summary, và 5 sample text boxes. Không commit ảnh hoặc JSONL lớn trong `data/`.

## Data Pipeline Utilities (`src/training/scripts/data_utils.py`)

### Augmentation Pipeline

- [x] Random crop, flip, color jitter, brightness
  - Trạng thái: Đã có preset Albumentations cho `detection`, `scene`, `ocr` và CLI `augment-preview --task ...`, mặc định vẫn dry-run. Chưa chạy augmentation trên dataset thật.
- [x] Resize + normalize chuẩn cho từng model
  - Trạng thái: Đã có preset resize/normalize ImageNet mean/std cho image tasks; action hiện chỉ có frame-sequence sampler, chưa train model.

### Split And Validation

- [x] Viết train/val splitter
  - Trạng thái: Đã có `stratified_split`, `group_aware_split`, CLI `split`, và tests.
- [x] Viết data validation script (check corrupt images, missing labels)
  - Trạng thái: Đã có YOLO validation check corrupt image, missing label/image, class id và bbox range. Đã thêm CLI `validate-video-manifest` để check file tồn tại, label membership, split/group leakage, và optional OpenCV video-open/frame-count check; manifest UCF-101 hiện validate `checked_files=2449 valid=True`.

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

- Hoàn thành scaffolding/config/docs/notebook exploration output/utility dry-run/test fixtures.
- Dataset public/local đã đồng bộ với config hiện tại: COCO train+val, UCF-101, Places365 val_256, TextOCR và MSVD đều có trong `data/external`; các output chính trong `data/processed` đã được tạo. `verify-dataset-paths --skip-downloads --include-outputs` hiện pass với `checked_files=26 valid=True`.
- Các mục còn mở đều là human-gated với dữ liệu custom thật: record/capture thực tế, annotate 200-500 ảnh, và ingest export YOLO custom thật bằng `normalize-custom-yolo` trước khi merge. Các tiện ích code-first như video frame-sequence loader, augmentation presets, `verify-dataset-paths`, `write-classes`, `build-scene-subset`, `merge-yolo`, `normalize-custom-yolo`, và `validate-video-manifest` đã có test.
