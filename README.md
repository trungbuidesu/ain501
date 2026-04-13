# ain501

AIN501 là dự án hỗ trợ phát triển ứng dụng AI và trợ lý hỗ trợ tiếp cận theo
hướng sạch, module hóa, dễ kiểm thử và dễ mở rộng.

## Cấu Trúc Dự Án

- `src/`: Mã nguồn chính. Tiện ích dữ liệu hiện chạy qua
  `python -m src.training.scripts...`.
- `configs/`: Cấu hình YAML/JSON.
- `configs/datasets/`: Cấu hình dataset cho Phase 0. Xem
  [configs/datasets/README.md](configs/datasets/README.md).
- `data/`: Dữ liệu local, archive tải về và artifact đã xử lý. Thư mục này
  không được commit vào git.
- `docs/`: Tài liệu dự án. Pipeline dữ liệu chính nằm ở
  [docs/data_pipeline.md](docs/data_pipeline.md).
- `notebooks/`: Notebook khởi động và khám phá cho từng dataset Phase 0.
- `tests/`: Kiểm thử đơn vị và kiểm thử tích hợp.
- `agents/`: Quy tắc làm việc, rà soát code và commit dành cho agent.

## Bắt Đầu

### 1. Yêu Cầu

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) hoặc
  [Anaconda](https://www.anaconda.com/).
- Python 3.10+.

### 2. Tạo Môi Trường

Repo hiện đang được kiểm tra trong conda env `trungbd`. Nếu env chưa có:

```bash
conda create --name trungbd python=3.10
conda activate trungbd
```

Nếu đã có env:

```bash
conda activate trungbd
```

### 3. Cài Đặt

Cài project ở chế độ editable kèm dependency phát triển:

```bash
pip install -e ".[dev]"
```

Nếu dùng Torch CPU/XPU, cài wheel phù hợp máy trước hoặc sau bước cài editable
theo hướng dẫn chính thức của PyTorch/Intel. Env local đã từng dùng
`torch 2.11.0+xpu`; máy không có GPU Intel Arc được hỗ trợ có thể bỏ qua kiểm
thử XPU.

Dự phòng PowerShell khi `python` hoặc `conda` không có trong PATH:

```powershell
$AIN501_PY="C:\Users\Admin\miniconda3\envs\trungbd\python.exe"
& $AIN501_PY -m pip install -e ".[dev]"
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

Các dataset lõi của Phase 0 hiện dùng:

- Phát hiện đối tượng: COCO 2017.
- Nhận diện hành động: UCF-101.
- Phân loại cảnh: Places365.
- Sinh mô tả video Tier 2/deferred: MSR-VTT.
- OCR: TextOCR.

## Quy Ước Phát Triển

Quy tắc commit và kiểm tra code chi tiết nằm trong
[agents/rules.md](agents/rules.md).

### Kiểm Tra Nhanh

- **Format**: `black .`
- **Lint**: `ruff check .`
- **Type check**: `mypy .`
- **Test**: `pytest`

Dự phòng PowerShell:

```powershell
$AIN501_PY="C:\Users\Admin\miniconda3\envs\trungbd\python.exe"
& $AIN501_PY -m black --check .
& $AIN501_PY -m ruff check .
& $AIN501_PY -m mypy .
& $AIN501_PY -m pytest -p no:cacheprovider
```
