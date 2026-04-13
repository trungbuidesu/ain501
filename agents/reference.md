# Tài liệu tham chiếu triển khai — AIN501

> Tài liệu này tổng hợp kiến trúc project, shared utilities, API reference, và các patterns chuẩn để tham chiếu khi triển khai tính năng mới.

---

## 1. Kiến trúc Project

```
ain501/
├── src/                    # Source code chính
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py     # export: TrainingLogger
│       └── logger.py       # Unified TensorBoard / W&B logger
├── models/                 # Model definitions, weights, checkpoints
├── configs/                # YAML / JSON config files
├── data/                   # Data directory (git-ignored)
├── tests/                  # Unit & integration tests
├── docs/                   # Documentation
├── agents/                 # Agent docs, rules, phase plans
├── pyproject.toml          # Dependencies + tool config
└── README.md
```

### Package install (editable mode)
```bash
conda activate ain501
pip install -e ".[dev]"
```
Sau khi install, import bằng `from src.utils import ...`

---

## 2. Môi trường & Phần cứng

| Item | Value |
|------|-------|
| Python | 3.10+ |
| Conda env | `ain501` |
| GPU | Intel Arc A770 |
| PyTorch | `2.11.0+xpu` (cài từ `--index-url https://download.pytorch.org/whl/xpu`) |
| Device string | `"xpu"` (**không** dùng `"cuda"`) |

### Device selection pattern (BẮT BUỘC)
```python
import torch

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

> ⚠️ **KHÔNG BAO GIỜ hardcode `"cuda"`** trong project này. Xem chi tiết: `agents/intel_arc_xpu_notes.md`

---

## 3. Shared Utilities Reference

### 3.1 TrainingLogger (`src/utils/logger.py`)

Unified logging interface cho TensorBoard và W&B. Dùng chung cho mọi training pipeline.

#### Import
```python
from src.utils import TrainingLogger
```

#### Khởi tạo
```python
logger = TrainingLogger(
    project="ain501-yolo",          # Tên project (W&B dùng)
    run_name="exp-001",             # Tên experiment
    backends=["tensorboard"],       # ["tensorboard"], ["wandb"], hoặc cả hai
    log_dir="runs",                 # Thư mục output (default: "runs")
    config={"lr": 1e-3, "epochs": 100},  # Hyperparameters
    tags=["baseline", "v1"],        # Tags (W&B only)
)
```

#### API Methods

| Method | Mô tả | TensorBoard | W&B |
|--------|--------|:-----------:|:---:|
| `log_scalar(tag, value, step)` | Log 1 scalar | ✅ | ✅ |
| `log_scalars(main_tag, {k: v}, step)` | Log nhiều scalar grouped | ✅ | ✅ |
| `log_image(tag, image, step)` | Log ảnh (Tensor CHW / numpy HWC) | ✅ | ✅ |
| `log_histogram(tag, values, step)` | Log histogram (weights, grads) | ✅ | ✅ |
| `log_text(tag, text, step)` | Log text | ✅ | ✅ |
| `log_artifact(path, type)` | Log file/dir as artifact | ❌ | ✅ |
| `log_model_graph(model, input)` | Log model architecture | ✅ | ✅ |
| `finish()` | Flush & close tất cả backends | ✅ | ✅ |

#### Ví dụ training loop đầy đủ
```python
import torch
import torch.nn as nn
from src.utils import TrainingLogger

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

with TrainingLogger(
    project="ain501",
    run_name="exp-001",
    backends=["tensorboard", "wandb"],
    config={"lr": 1e-3, "batch_size": 32, "device": str(device)},
) as logger:

    for epoch in range(num_epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.log_scalar("train/loss", train_loss, step=epoch)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.log_scalars("val", {"loss": val_loss, "acc": val_acc}, step=epoch)

        # Log weights histogram mỗi 10 epochs
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                logger.log_histogram(f"weights/{name}", param.data.cpu(), step=epoch)

    # Lưu model & log artifact
    torch.save(model.state_dict(), "models/best_model.pt")
    logger.log_artifact("models/best_model.pt", type="model")
```

#### Xem kết quả TensorBoard
```bash
tensorboard --logdir runs/
# Mở http://localhost:6006
```

---

## 4. Dependencies chính

### Deep Learning & Vision
| Package | Mục đích | Docs |
|---------|----------|------|
| `torch` + `torchvision` + `torchaudio` | Core DL framework | https://pytorch.org/docs/ |
| `ultralytics` | YOLOv8 object detection | https://docs.ultralytics.com/ |
| `onnx` + `onnxruntime` | Model export & inference | https://onnxruntime.ai/docs/ |
| `opencv-python` | Image/video processing | https://docs.opencv.org/ |
| `Pillow` | Image I/O | https://pillow.readthedocs.io/ |

### OCR & Pose
| Package | Mục đích | Docs |
|---------|----------|------|
| `paddleocr` | OCR (text detection + recognition) | https://paddlepaddle.github.io/PaddleOCR/ |
| `mediapipe` | Pose estimation, hand tracking | https://ai.google.dev/edge/mediapipe |

### Vector Search & NLP
| Package | Mục đích | Docs |
|---------|----------|------|
| `faiss-cpu` | Vector similarity search | https://github.com/facebookresearch/faiss |
| `sentence-transformers` | Text embeddings | https://www.sbert.net/ |

### GUI & Screen Capture
| Package | Mục đích | Docs |
|---------|----------|------|
| `PySide6` | Qt6 GUI framework | https://doc.qt.io/qtforpython-6/ |
| `dxcam` | DirectX screen capture (Windows) | https://github.com/ra1nty/DXcam |
| `mss` | Cross-platform screenshot | https://python-mss.readthedocs.io/ |

### Logging & Monitoring
| Package | Mục đích | Docs |
|---------|----------|------|
| `tensorboard` | Local training visualization | https://www.tensorflow.org/tensorboard |
| `wandb` | Cloud experiment tracking | https://docs.wandb.ai/ |

---

## 5. Coding Patterns

### 5.1 Config loading
```python
import yaml
from pathlib import Path

def load_config(name: str) -> dict:
    path = Path("configs") / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

### 5.2 Device-agnostic training
```python
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Mixed precision trên XPU
with torch.autocast(device_type="xpu", dtype=torch.float16):
    output = model(input_tensor)
```

### 5.3 Model save/load
```python
# Save
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, "models/checkpoint.pt")

# Load
checkpoint = torch.load("models/checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

### 5.4 ONNX export
```python
dummy_input = torch.randn(1, 3, 640, 640, device=device)
torch.onnx.export(model, dummy_input, "models/model.onnx", opset_version=17)
```

---

## 6. Tham chiếu nhanh các file trong `agents/`

| File | Nội dung |
|------|----------|
| `phase0.md` | Checklist khởi tạo project |
| `rules.md` | Quy tắc commit (Conventional Commits) + check rules (ruff, black, mypy, pytest) |
| `intel_arc_xpu_notes.md` | Hướng dẫn sử dụng Intel Arc A770, cài đặt PyTorch XPU, lưu ý quan trọng |
| `reference.md` | **(file này)** Tài liệu tham chiếu tổng hợp |

---

## 7. Checklist khi triển khai tính năng mới

- [ ] Tạo branch theo convention: `feat/<tên>`, `fix/<tên>`, `refactor/<tên>`
- [ ] Device selection: dùng `"xpu"` pattern, **không** hardcode `"cuda"`
- [ ] Logging: dùng `TrainingLogger` từ `src/utils` (không tự viết logging riêng)
- [ ] Config: đặt file YAML trong `configs/`
- [ ] Tests: viết test kèm trong `tests/test_<tên>.py`
- [ ] Linting: chạy `ruff check .` và `black .` trước khi commit
- [ ] Type hints: tất cả public functions phải có type annotations
- [ ] Commit message: theo Conventional Commits (xem `agents/rules.md`)
