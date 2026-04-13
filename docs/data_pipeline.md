# Phase 0 Data Pipeline

Tài liệu này mô tả pipeline dữ liệu cho accessibility assistant. Mặc định mọi
lệnh đều ưu tiên dry-run để xem trước thao tác, tránh tải dataset lớn hoặc ghi
file ngoài ý muốn.

## Cấu Trúc Thư Mục

- `configs/datasets/`: Cấu hình dataset và model data có version.
- `data/downloads/`: Archive tải thủ công hoặc tải bằng script opt-in.
- `data/external/`: Dataset bên thứ ba sau khi giải nén.
- `data/custom/raw_frames_private/`: Frame capture thô có thể chứa thông tin
  riêng tư; chỉ giữ local.
- `data/custom/sanitized_frames/`: Frame đã được review và lọc riêng tư.
- `data/custom/annotation_exports/`: Export từ CVAT hoặc Roboflow.
- `data/processed/`: Dataset đã convert, split và merge để train.
- `data/tts_validation/`: File WAV sinh ra khi validate Piper TTS.
- `models/voices/piper/`: File voice Piper local gồm `.onnx` và `.onnx.json`.
- `models/depth/midas/`: Slot dự phòng cho artifact depth-estimation model.

`data/` đang được git-ignore. Không commit raw capture, dataset bên thứ ba,
label đã convert, audio sinh ra, hoặc model weights.

## Mặc Định Dry-Run

Các thao tác lớn hoặc có side effect đều phải opt-in:

```bash
python -m training.scripts.data_utils download-plan
python -m training.scripts.data_utils build-action-manifest
python -m training.scripts.data_utils convert --coco-json path/to/instances.json --image-root path/to/images --output-root data/processed/demo --classes classes.txt
python -m training.scripts.data_utils merge-yolo --sources data/custom/export --output-root data/processed/merged --classes classes.txt --dry-run
python -m training.scripts.data_utils augment-preview --input-dir path/to/images --output-dir data/processed/augment_preview --task detection
python -m training.scripts.data_utils validate-tts --dry-run
python -m training.scripts.capture_frames --duration-seconds 3600 --fps 1
```

Fallback PowerShell khi `python` không có trong PATH:

```powershell
$AIN501_PY="C:\Users\bdtrung29.1\miniconda3\envs\ain501\python.exe"
& $AIN501_PY -m training.scripts.data_utils download-plan
& $AIN501_PY -m training.scripts.data_utils build-action-manifest
& $AIN501_PY -m training.scripts.data_utils validate-tts --dry-run
& $AIN501_PY -m training.scripts.capture_frames --duration-seconds 60 --fps 1
```

Chỉ dùng `--execute` sau khi đã xác nhận path, dung lượng lưu trữ và xử lý
riêng tư.

## Các Bước Dataset Thủ Công

- COCO 2017: tải `train2017.zip`, `val2017.zip` và
  `annotations_trainval2017.zip` từ trang COCO chính thức vào
  `data/downloads/coco/`, rồi giải nén vào `data/external/coco2017/`.
- HMDB-51: tải video và split train/test chính thức từ trang dataset vào
  `data/external/hmdb51/`.
- Places365: dùng biến thể Places365 256px và category file dưới
  `data/external/places365/`.
- MSR-VTT: tải theo điều khoản phân phối chính thức và normalize annotation về
  dạng `video_id -> captions[]`.
- ICDAR 2015: đăng ký và tải từ Robust Reading Competition, sau đó validate nội
  dung zip local trước khi xử lý.
- Piper TTS: đặt voice files dưới `models/voices/piper/` theo
  `configs/datasets/tts_piper_accessibility.yaml`.

## Code-First Utilities

- Action recognition: `build-action-manifest` đọc HMDB-style class folders, optional
  official split files và chỉ ghi manifest khi có `--execute`.
- Frame sequence: sampler dùng mặc định `frame_sequence_length=16` và
  `frame_stride=2` theo action config; video thiếu frame được bỏ qua ở sequence
  manifest.
- Image augmentation: `augment-preview` hỗ trợ `--task detection|scene|ocr` và
  `--size`; dry-run mặc định chỉ đếm ảnh, `--execute` mới ghi preview.
- Packaging: package editable cần expose cả `src*` và `training*` để các lệnh
  `python -m training.scripts...` chạy được ngoài repo root.

## Env Ghi Nhận

- Dev env đã dùng: conda env `trungbd`.
- PyTorch XPU wheel: `torch 2.11.0+xpu`, `torchvision 0.26.0+xpu`,
  `torchaudio 2.11.0+xpu`.
- Máy hiện detect `Intel(R) UHD Graphics 730` qua XPU nhưng PyTorch cảnh báo đây
  không phải Intel Arc-supported GPU; XPU training sanity test sẽ skip nếu không
  có Intel Arc.

## Workflow CVAT Hoặc Roboflow

1. Capture frame thô vào `data/custom/raw_frames_private/`.
2. Review và loại bỏ frame nhạy cảm.
3. Làm mờ mặt người và biển số xe trước khi annotate.
4. Chỉ copy frame đã sanitize sang `data/custom/sanitized_frames/`.
5. Annotate 200-500 ảnh bằng canonical class names trong object detection
   config.
6. Export theo format Ultralytics YOLO.
7. Validate và preview remap trước khi ghi merged output:

```bash
python -m training.scripts.data_utils validate --dataset-root data/custom/annotation_exports/export_001 --classes data/processed/object_detection_accessibility/classes.txt
python -m training.scripts.data_utils merge-yolo --sources data/custom/annotation_exports/export_001 data/processed/object_detection_accessibility --output-root data/processed/object_detection_accessibility_merged --classes data/processed/object_detection_accessibility/classes.txt --dry-run
```

## Checklist Riêng Tư

- Chỉ capture những scenario cần cho label Phase 0.
- Tránh nhà riêng và không gian nội bộ của doanh nghiệp nếu chưa có consent
  liên quan.
- Với capture ngoài trời hoặc nơi công cộng, làm mờ mặt người và biển số xe
  trước khi annotate/export. Phase 0 không cần định danh cá nhân.
- Không ghi âm cho vision datasets.
- Xóa frame có tài liệu cá nhân đọc được, thẻ thanh toán, thông tin sức khỏe,
  địa chỉ, số điện thoại, chat riêng tư hoặc credential.
- Giữ raw capture local trong `data/custom/raw_frames_private/`; chỉ frame đã
  sanitize mới được đưa sang bước annotation.
- Manifest cần có các trường: ngày capture, loại địa điểm, trạng thái consent,
  trạng thái sanitize, người review và ghi chú.
- Xóa frame bị reject hoặc raw private frame sau khi subset đã sanitize được tạo
  và backup theo policy local.

## Lệnh Acceptance

```bash
python -m training.scripts.data_utils download-plan
python -m training.scripts.data_utils build-action-manifest
python -m training.scripts.data_utils validate-tts --dry-run
python -m training.scripts.capture_frames --duration-seconds 60 --fps 1
ruff check .
black --check .
mypy .
pytest
```

Nếu shell không nhận `python`, dùng fallback `$AIN501_PY` ở trên cho các lệnh
tương đương.

Phase 0 không bị block bởi action modeling cho “approaching”. “Approaching” là
extension Phase 9, sẽ derive từ object detection và tracking.
