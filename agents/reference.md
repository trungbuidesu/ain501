# Tài Liệu Tham Chiếu Triển Khai - AIN501

Tài liệu này là bản tham chiếu nhanh cho agent/engineer khi làm việc trong
repo. Nội dung được chuẩn hóa theo [README.md](../README.md).

## 1. Kiến Trúc Project

```text
ain501/
├── src/                    # Mã nguồn chính
│   ├── training/
│   │   ├── data/           # Pipeline dữ liệu theo từng miền
│   │   ├── models/         # Model/backbone training utilities
│   │   └── scripts/        # CLI entrypoints
│   └── utils/
│       ├── logger.py       # Compatibility shim cho TrainingLogger
│       └── loggers/        # TensorBoard / W&B logger implementation
├── configs/                # YAML/JSON config files
├── data/                   # Dữ liệu local, git-ignored
├── docs/                   # Tài liệu dự án
├── notebooks/              # Notebook khám phá dataset Phase 0
├── tests/                  # Unit/integration tests
└── agents/                 # Rule, reference và ghi chú vận hành
```

Pipeline dữ liệu chạy qua CLI thống nhất:

```bash
python -m src.training.scripts.data_utils download-plan
python -m src.training.scripts.data_utils verify-dataset-paths --skip-downloads
python -m src.training.scripts.data_utils validate-tts --dry-run
```

Các model utilities hiện có:

```bash
python -m src.training.scripts.mobilenetv3 describe
python -m src.training.scripts.yolov8n describe
```

## 2. Môi Trường Và Phần Cứng

Mặc định làm theo README: dùng Miniconda/Anaconda, Python 3.10+ và cài dev
extra.

```bash
conda activate <env-name>
pip install -e ".[dev]"
```

Yêu cầu môi trường chuẩn:

| Item | Value |
| --- | --- |
| Conda env | `<env-name>` (tự quy định khi khởi tạo) |
| Python | `3.10+` |
| XPU Support | Hỗ trợ natively qua `torch.xpu` (ví dụ `Intel(R) Arc(TM) A770 Graphics`) |

Ghi chú Arc-specific, xem [intel_arc_xpu_notes.md](intel_arc_xpu_notes.md).

Device selection trong code nên dùng fallback `xpu -> cpu`, không hardcode
`cuda`:

```python
import torch

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model = model.to(device)
batch = batch.to(device)
```

## 3. Shared Utilities

### TrainingLogger

Import chuẩn:

```python
from src.utils import TrainingLogger
```

`src.utils.logger.TrainingLogger` vẫn là shim tương thích import path cũ; logic
chính nằm trong `src.utils.loggers.training.TrainingLogger`.

Ví dụ tối thiểu:

```python
with TrainingLogger(
    project="ain501",
    run_name="exp-001",
    backends=["tensorboard"],
    log_dir="runs",
    config={"lr": 1e-4, "batch_size": 32},
) as logger:
    logger.log_scalar("train/loss", 0.5, step=1)
```

Metrics training chính thức nên dùng tag nhóm như `train/loss`, `val/loss`,
`val/accuracy`, `val/mAP50`, `val/mAP50-95`, và `system/epoch_time_s`.

## 4. Dataset Và Model State

Dataset Phase 0 được cấu hình trong `configs/datasets/` và ghi tiến độ ở
`checklist/phase0_data_pipeline.md`. Dữ liệu thật nằm dưới `data/` và không được
commit.

Model configs hiện có trong `configs/models/`:

- `mobilenetv3_small.yaml`: shared feature extractor MobileNetV3-Small.
- `yolov8n.yaml`: config train/evaluate dry-run-safe cho YOLOv8n.

YOLOv8n training thật chỉ nên chạy sau khi YOLO train/val processed đã tồn tại
và data YAML trỏ đúng `images/train` + `images/val`.

## 5. Quy Tắc Kiểm Tra

Theo [rules.md](rules.md), trước khi commit cần chạy:

```bash
python -m black --check .
python -m ruff check .
python -m mypy .
python -m pytest -q
```

Với PowerShell và env đã activate:

```powershell
$AIN501_PY=Join-Path $env:CONDA_PREFIX "python.exe"
& $AIN501_PY -m black --check .
& $AIN501_PY -m ruff check .
& $AIN501_PY -m mypy .
& $AIN501_PY -m pytest -q
```

Commit message theo Conventional Commits: `<type>(<scope>): <description>`.

## 6. Checklist Khi Triển Khai Tính Năng

- Dùng import path `src.*` và CLI `python -m src.training.scripts...`.
- Không ghi artifact nặng vào git: `data/`, `models/`, `runs/`, `reports/`.
- Dùng `TrainingLogger` cho mọi training run thay vì logging tự phát.
- Không hardcode `cuda`; dùng device fallback `xpu -> cpu`.
- Cập nhật checklist/docs khi trạng thái dữ liệu hoặc model artifact thay đổi.
- Chạy black, ruff, mypy, pytest và các validation dataset liên quan.
