# ain501

AIN501 là dự án hỗ trợ phát triển ứng dụng AI và trợ lý hỗ trợ tiếp cận theo
hướng sạch, module hóa, dễ kiểm thử và dễ mở rộng.

Phase 0 tập trung pipeline dữ liệu đa tác vụ: mã trong `src/`, cấu hình YAML dưới
`configs/datasets/`, gọi CLI thống nhất `python -m src.training.scripts...`.
Metadata package, dependency và extras `[dev]`, `[tts]` khai báo trong
[pyproject.toml](pyproject.toml) (`requires-python >=3.10`).

## Cấu Trúc Dự Án

- `src/`: Mã nguồn chính. Tiện ích dữ liệu hiện chạy qua
  `python -m src.training.scripts...`.
- `configs/`: Cấu hình YAML/JSON.
- `configs/datasets/`: Cấu hình dataset cho Phase 0. Xem
  [configs/datasets/README.md](configs/datasets/README.md).
- `data/`: Dữ liệu local, archive tải về và artifact đã xử lý. Thư mục này
  không được commit vào git.
- `models/`: Artifact nặng cục bộ (Piper `.onnx`, slot depth). Cấu trúc thư mục
  và quy ước lưu file: [docs/data_pipeline.md](docs/data_pipeline.md).
- `docs/`: Tài liệu dự án. Pipeline dữ liệu chính nằm ở
  [docs/data_pipeline.md](docs/data_pipeline.md).
- `notebooks/`: Notebook khởi động và khám phá cho từng dataset Phase 0.
- `checklist/`: Theo dõi tiến độ thực tế theo từng task dữ liệu
  ([checklist/phase0_data_pipeline.md](checklist/phase0_data_pipeline.md)).
- `tests/`: Kiểm thử đơn vị và kiểm thử tích hợp.
- `agents/`: Quy tắc làm việc, rà soát code và commit dành cho agent. Bản đồ
  dependency nhanh: [agents/reference.md](agents/reference.md).

## Bắt Đầu

### 1. Yêu Cầu

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) hoặc
  [Anaconda](https://www.anaconda.com/).
- Python 3.10+.

### 2. Tạo Môi Trường

Tạo một conda env riêng cho project. Thay `<env-name>` bằng tên env bạn muốn dùng:

```bash
conda create --name <env-name> python=3.10
conda activate <env-name>
```

Nếu đã có sẵn env phù hợp, chỉ cần activate env đó:

```bash
conda activate <env-name>
```

### 3. Cài Đặt

Cài project ở chế độ editable kèm dependency phát triển. Extra `[dev]` bao gồm
`black[jupyter]` để `black .` format được cả Python files và notebook code cells:

```bash
pip install -e ".[dev]"
```

Tùy chọn Piper TTS (cài `piper-tts` qua extra `[tts]` trong `pyproject.toml`):

```bash
pip install -e ".[tts]"
```

Nếu dùng Torch CPU/XPU, cài wheel phù hợp máy trước hoặc sau bước cài editable
theo hướng dẫn chính thức của PyTorch/Intel. Máy không có GPU Intel Arc được hỗ
trợ có thể bỏ qua kiểm thử XPU. Ghi chú phần cứng và XPU:
[agents/intel_arc_xpu_notes.md](agents/intel_arc_xpu_notes.md).

Dự phòng PowerShell khi `python` không có trong PATH nhưng env đã được activate:

```powershell
$AIN501_PY=Join-Path $env:CONDA_PREFIX "python.exe"
& $AIN501_PY -m pip install -e ".[dev]"
# Tùy chọn: & $AIN501_PY -m pip install -e ".[tts]"
```

## Pipeline Dữ Liệu

Các cấu hình dataset nằm trong [configs/datasets](configs/datasets) và được mô
tả ở [configs/datasets/README.md](configs/datasets/README.md). Mặc định các
lệnh dataset ưu tiên dry-run để tránh tải/ghi dữ liệu lớn ngoài ý muốn.

Các lệnh xem trạng thái nhanh:

```bash
python -m src.training.scripts.data_utils download-plan
python -m src.training.scripts.data_utils verify-dataset-paths --tasks object_detection action_recognition scene_classification --skip-downloads
python -m src.training.scripts.data_utils validate-tts --dry-run
```

### Lệnh con `data_utils` (CLI)

Module [src/training/scripts/data_utils.py](src/training/scripts/data_utils.py)
đăng ký các subcommand (mỗi lệnh có `--help`).

| Subcommand | Vai trò |
| --- | --- |
| `download-plan` | In kế hoạch URL/path tải từ YAML trong `configs/datasets/`. |
| `verify-dataset-paths` | Kiểm tra cấu trúc thư mục local theo các `task` đã cấu hình. |
| `write-classes` | Ghi `classes.txt` chuẩn từ subset class trong config object detection. |
| `split` | Chia tập (stratified / group-aware) cho manifest hoặc danh sách file. |
| `validate` | Kiểm tra layout YOLO: ảnh/nhãn, class id, phạm vi bbox. |
| `convert` | Chuyển COCO JSON (bbox xywh pixel) sang nhãn YOLO đã chuẩn hóa. |
| `merge-yolo` | Gộp nhiều thư mục YOLO; mặc định dry-run, dùng `--execute` khi ghi file. |
| `augment-preview` | Xem trước augmentation (Albumentations) theo `--task`; mặc định dry-run. |
| `build-action-manifest` | Tạo manifest video / chuỗi frame cho thư mục class kiểu UCF-101. |
| `build-scene-subset` | Tạo subset cân bằng Places365 từ filelist hoặc thư mục ảnh. |
| `validate-tts` | Kiểm tra cấu hình Piper và WAV; bỏ `--dry-run` khi cần ghi/kiểm âm thanh. |

### Ghi màn hình (dữ liệu tùy chỉnh)

[src/training/scripts/capture_frames.py](src/training/scripts/capture_frames.py)
ưu tiên `dxcam` (Windows / DirectX), fallback `mss`. Không truyền `--execute` thì
chỉ ước lượng số frame; có `--execute` mới ghi JPEG. Mặc định `--output-dir` là
`data/custom/raw_frames`. Quy trình privacy và bước sau capture (sanitize, export
nhãn): [docs/data_pipeline.md](docs/data_pipeline.md).

```bash
python -m src.training.scripts.capture_frames --duration-seconds 60 --fps 1
python -m src.training.scripts.capture_frames --duration-seconds 60 --fps 1 --execute
```

Các bộ dữ liệu (dataset) đang được khai báo cấu hình:

- **Phát hiện đối tượng (Object Detection):** Bộ gốc COCO 2017. Cấu trúc tập class thiết yếu để huấn luyện mạng theo định dạng YOLO.
- **Nhận diện hành động (Action Recognition):** Bộ gốc UCF-101. Lấy mẫu luồng frame video nhận diện chuỗi hành động có sẵn và tự định nghĩa.
- **Phân loại không gian (Scene Classification):** Bộ gốc Places365. Subset ảnh 256px đi kèm cơ chế tự động cân bằng cho các không gian bối cảnh.
- **Trích xuất văn bản (OCR Scene Text):** Nhắm tới nền tảng TextOCR. Đọc chữ trên các vật thể và biển báo tự nhiên.
- **Sinh mô tả video (Video Captioning):** Bộ gốc MSVD. Sinh caption thuyết minh sự kiện (Tier 2/deferred).
- **Giọng nói ảo (TTS Validation):** Khai phá các model ONNX của Piper để kiểm chứng tín hiệu phát âm thanh đầu ra.
- **Ước lượng chiều sâu (Depth Estimation):** Slot model dự phòng cho thuật toán MiDaS.

## Notebook khám phá

| Notebook | Mục tiêu |
| --- | --- |
| `notebooks/0.3_coco_detection_explore.ipynb` | COCO / phát hiện đối tượng |
| `notebooks/0.4_action_ucf101_explore.ipynb` | UCF-101 / nhận diện hành động |
| `notebooks/0.5_places_scene_subset.ipynb` | Places365 / phân loại cảnh |
| `notebooks/0.6_msvd_captioning_explore.ipynb` | MSVD / mô tả video (Tier 2) |
| `notebooks/0.7_textocr_explore.ipynb` | TextOCR / chữ trong cảnh |
| `notebooks/0.8_tts_piper_validation.ipynb` | Piper TTS / giọng ONNX |
| `notebooks/0.9_depth_midas_reserved.ipynb` | Độ sâu (MiDaS, dự phòng) |

## Quy Ước Phát Triển

Quy tắc commit và kiểm tra code chi tiết nằm trong
[agents/rules.md](agents/rules.md).

### Kiểm Tra Nhanh

- **Format**: `black .` (format cả `.py` và `.ipynb` code cells; cần `black[jupyter]` từ extra `[dev]`)
- **Lint**: `ruff check .`
- **Type check**: `mypy .`
- **Test**: `pytest -p no:cacheprovider`

[tests/test_xpu_training.py](tests/test_xpu_training.py) chỉ chạy khi
`torch.xpu.is_available()` và tên thiết bị chứa `arc` (Intel Arc); các môi trường
khác pytest sẽ `skip` test đó.

Dự phòng PowerShell:

```powershell
$AIN501_PY=Join-Path $env:CONDA_PREFIX "python.exe"
& $AIN501_PY -m black --check .
& $AIN501_PY -m ruff check .
& $AIN501_PY -m mypy .
& $AIN501_PY -m pytest -p no:cacheprovider
```

## Theo dõi tiến độ

Chi tiết trạng thái dataset và task:
[checklist/phase0_data_pipeline.md](checklist/phase0_data_pipeline.md).

## License

Xem [LICENSE](LICENSE).
