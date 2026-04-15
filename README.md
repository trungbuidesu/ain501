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
- `src/training/data/`: Module triển khai pipeline dữ liệu theo từng miền
  (COCO/YOLO, UCF-101, Places365, TextOCR, MSVD, Piper TTS, augmentation,
  split/validation). `src/training/scripts/data_utils.py` là CLI facade giữ
  import path cũ.
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
- `checklist/`: Theo dõi tiến độ thực tế theo từng task dữ liệu và model
  ([checklist/phase0_data_pipeline.md](checklist/phase0_data_pipeline.md)).
- `tests/`: Kiểm thử đơn vị và kiểm thử tích hợp.
- `agents/`: Quy tắc làm việc, rà soát code và commit dành cho agent. Bản đồ
  dependency nhanh: [agents/reference.md](agents/reference.md).

## Xem Tài Liệu (MkDocs)

Tài liệu web dùng [MkDocs](https://www.mkdocs.org/) với theme
[Material](https://squidfunk.github.io/mkdocs-material/). Cấu hình nằm ở
[mkdocs.yml](mkdocs.yml); nội dung Markdown trong [docs/](docs/).

Cài gói qua dev extra của project:

```bash
pip install -e ".[dev]"
```

Từ thư mục gốc repo:

- **Xem local:** `mkdocs serve` — mở trình duyệt tại địa chỉ hiển thị (thường
  `http://127.0.0.1:8000`).
- **Build tĩnh:** `mkdocs build` — output vào `site/` (thư mục này đã được
  git-ignore).

Tài liệu vận hành pipeline chi tiết vẫn có thể đọc trực tiếp file
[docs/data_pipeline.md](docs/data_pipeline.md) trên Git.

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

### Quy ước thiết bị (train vs production)

- **Train / fine-tune:** có thể dùng `--device auto` để tận dụng tăng tốc phần cứng
  (Intel Arc XPU nếu khả dụng, hoặc CUDA nếu môi trường hỗ trợ).
- **Production / benchmark ONNX:** pipeline benchmark tuần 6 mặc định chạy
  **CPU** qua `CPUExecutionProvider` trong
  [configs/benchmark/week6.yaml](configs/benchmark/week6.yaml).
- **Giữ `auto` trong production:** vẫn hợp lệ nếu muốn tận dụng phần cứng hiện có.
  Khi cần kết quả lặp lại ổn định giữa máy (reproducible CPU), ép CPU tường minh:
  `--cpu` hoặc `--device cpu` (ví dụ với
  `python -m src.training.scripts.verify_prebuilt_week5 ...`).

Dự phòng PowerShell khi `python` không có trong PATH nhưng env đã được activate:

```powershell
$AIN501_PY=Join-Path $env:CONDA_PREFIX "python.exe"
& $AIN501_PY -m pip install -e ".[dev]"
# Tùy chọn: & $AIN501_PY -m pip install -e ".[tts]"
```

### App runtime (object-to-speech, optional Tier 1 router)

- **Chạy mặc định:** `python -m src.app` (config: [`configs/app.yaml`](configs/app.yaml))
- **Tier 1:** bật `tier1.enabled: true` trong [`configs/app.yaml`](configs/app.yaml) (cần ONNX scene + MoViNet; xem `models/registry.json`). Tài liệu: [docs/training/tier1_pipeline.md](docs/training/tier1_pipeline.md).
- **Chỉ YOLO (debug):** `python -m src.app --object-only`
- **Benchmark CPU từng agent Tier 1:** `python -m src.training.scripts.benchmark_tier1_agents` → `reports/phase3/tier1_benchmark.md`
- **RAG Lite:** bật `rag.enabled: true` trong [`configs/app.yaml`](configs/app.yaml) (cần `models/minilm_l6.onnx` + `data/knowledge/*.json`). Tài liệu: [docs/training/rag_orchestrator.md](docs/training/rag_orchestrator.md).

**Môi trường:** khuyến nghị conda env `ain501` (Python 3.10+), cài dev: `pip install -e ".[dev]"`.

**Kiểm tra đầy đủ trước khi PR (từ thư mục gốc repo):** `black .` → `ruff check .` → `mypy .` → `pytest -q` → `python -m src.app --dry-run` → `python -m src.app --visual --dry-run`. Không cần `--config` trừ khi muốn file YAML khác; mặc định là [`configs/app.yaml`](configs/app.yaml).

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
đăng ký các subcommand (mỗi lệnh có `--help`) và gọi implementation trong
[src/training/data](src/training/data).

| Subcommand | Vai trò |
| --- | --- |
| `download-plan` | In kế hoạch URL/path tải từ YAML trong `configs/datasets/`. |
| `verify-dataset-paths` | Kiểm tra cấu trúc thư mục local theo các `task` đã cấu hình. |
| `write-classes` | Ghi `classes.txt` chuẩn từ subset class trong config object detection. |
| `split` | Chia tập (stratified / group-aware) cho manifest hoặc danh sách file. |
| `validate` | Kiểm tra layout YOLO: ảnh/nhãn, class id, phạm vi bbox. |
| `convert` | Chuyển COCO JSON (bbox xywh pixel) sang nhãn YOLO đã chuẩn hóa. |
| `merge-yolo` | Gộp nhiều thư mục YOLO; mặc định dry-run, dùng `--execute` khi ghi file. |
| `normalize-custom-yolo` | Chuẩn hóa export YOLO từ Roboflow/CVAT/Ultralytics vào layout `images/<split>` + `labels/<split>`; mặc định dry-run, dùng `--execute` khi ghi file. |
| `augment-preview` | Xem trước augmentation (Albumentations) theo `--task`; mặc định dry-run. |
| `build-action-manifest` | Tạo manifest video / chuỗi frame cho thư mục class kiểu UCF-101. |
| `validate-video-manifest` | Kiểm tra manifest video: file tồn tại, label hợp lệ, split/group leakage, và optional OpenCV frame-count check. |
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
- **Kho Tri Thức Vector (Vector Database):** Tích hợp ChromaDB để lưu trữ không gian nhúng (Embeddings) phục vụ truy xuất ngữ nghĩa (RAG) và tra cứu siêu dữ liệu dự án.

## MVP pipeline (Object-to-Speech)

Ứng dụng chạy pipeline capture → YOLO → mô tả không gian (template / RAG) → Piper TTS. Hướng dẫn đầy đủ (cài Piper/giọng, chỉnh [`configs/app.yaml`](configs/app.yaml) và `capture_pipeline.yaml`, lệnh chạy, đầu ra `reports/app/`, xử lý sự cố) nằm trong
[docs/training/app_runtime.md](docs/training/app_runtime.md).

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

Checklist model shared feature extractor:
[checklist/mobilenetv3_small.md](checklist/mobilenetv3_small.md).

Checklist object detection model:
[checklist/yolov8n.md](checklist/yolov8n.md).

## License

Xem [LICENSE](LICENSE).
