# Checklist Phase 0 - Data Pipeline Accessibility Assistant

File này dùng để theo dõi tiến độ Phase 0. Quy ước:

- `[x]` đã implement trong repo hoặc đã hoàn thành thực tế.
- `[ ]` chưa hoàn thành.
- `Trạng thái`: ghi chú ngắn để biết đang ở mức scaffold, dry-run hay đã chạy dữ liệu thật.

## Object Detection Dataset

- [ ] Download COCO 2017 train + val (~20GB)
  - Trạng thái: Đang làm một phần trên dữ liệu thật. Đã tải/extract `annotations_trainval2017.zip` và `val2017.zip`; đã ghi `classes.txt`, convert COCO val -> YOLO với 3445 ảnh selected và validate YOLO `checked_files=6890 valid=True`. `train2017.zip` partial do lượt tải bị ngắt đã được xóa; `train2017.zip`/`train2017` hiện chưa có nên mục tổng thể còn mở.
- [x] Hiểu COCO annotation format (JSON, bbox, categories)
  - Trạng thái: Đã implement parser/converter COCO bbox `[x, y, width, height]` sang YOLO trong `src/training/scripts/data_utils.py`, có test fixture.
- [ ] Viết data exploration notebook: phân tích distribution classes, bbox sizes
  - Trạng thái: Notebook starter `notebooks/0.3_coco_detection_explore.ipynb` đã có fallback fixture để chạy khi chưa có COCO local. Chưa chạy phân tích đầy đủ trên COCO train+val thật.

### Optional - Custom Domain Data

- [ ] Record screen 30-60 phút với các scenario khác nhau
  - Trạng thái: Chưa record thật. Đã có script dry-run `src/training/scripts/capture_frames.py`.
- [ ] Annotate bằng CVAT hoặc Roboflow (200-500 images)
  - Trạng thái: Chưa annotate thật. Đã có workflow trong `docs/data_pipeline.md`.
- [ ] Convert annotation sang YOLO format
  - Trạng thái: Đã có COCO -> YOLO converter và YOLO validator trong `src/training/scripts/data_utils.py`. Chưa có converter CVAT/Roboflow raw export -> YOLO hoàn chỉnh và chưa chạy với export thật.
- [x] Merge với subset COCO nếu cần
  - Trạng thái: Đã implement `merge-yolo --dry-run` để preview remap trước khi ghi file. Chưa merge dataset thật.

## Action Recognition Dataset

- [ ] Download UCF-101
  - Trạng thái: Chưa download dữ liệu thật. Bootstrap action đã chuyển hẳn sang UCF-101 trong `configs/datasets/action_accessibility.yaml`; đặt video dưới `data/external/ucf101/UCF-101` và official split files dưới `data/external/ucf101/ucfTrainTestlist`.
- [x] Viết data loader: video -> frame sequences
  - Trạng thái: Đã implement action manifest builder cho class-folder videos, official split parser cho UCF-101, frame-sequence sampler và video manifest validation trong `src/training/scripts/data_utils.py`, có test fixture. Chưa chạy trên UCF-101 thật.
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
- Chưa hoàn thành đầy đủ download dataset thật; hiện có COCO annotations + val2017 local, val YOLO output đã validate, Places365 devkit + val_256 local và scene manifest balanced 1600 rows. Vẫn chưa có COCO train2017, UCF-101 local, capture/annotate thật, converter CVAT/Roboflow raw export -> YOLO, và analysis notebook chạy trên dataset thật đầy đủ.
- Hai phần code ưu tiên trước đó đã được bổ sung ở mức code-first: video frame-sequence loader và augmentation presets. Bổ sung thêm core dataset run helpers: `verify-dataset-paths`, `write-classes`, `build-scene-subset`; action manifest hiện target UCF-101. Ưu tiên tiếp theo là tải/đặt dataset thật vào `data/external` rồi chạy pipeline execute.
