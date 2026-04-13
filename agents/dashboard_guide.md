# Hướng dẫn sử dụng Dashboard — TensorBoard & Weights & Biases

> **Quy định**: Mọi training run trong project **BẮT BUỘC** phải log metrics qua `TrainingLogger`.  
> Không được train "mù" — không log, không dashboard.

---

## 1. Khi nào dùng backend nào?

| Tình huống | Backend | Lý do |
|------------|---------|-------|
| Debug nhanh, train local | `tensorboard` | Offline, không cần tài khoản, nhanh |
| Experiment tracking dài hạn | `wandb` | Cloud dashboard, so sánh runs, team sharing |
| So sánh nhiều runs phức tạp | `wandb` | Bảng so sánh, sweep, auto-charts |
| Demo / báo cáo cho team | `wandb` | Link chia sẻ, report builder |
| Train chính thức | `tensorboard` + `wandb` | Cả hai — local backup + cloud tracking |

### Quy tắc chọn backend

```python
# Dev / debug nhanh
logger = TrainingLogger(backends=["tensorboard"])

# Experiment chính thức
logger = TrainingLogger(backends=["tensorboard", "wandb"])

# KHÔNG BAO GIỜ train mà không có logger
# ❌ SAI:
for epoch in range(100):
    loss = train(...)
    print(f"loss: {loss}")  # Không đủ!

# ✅ ĐÚNG:
with TrainingLogger(...) as logger:
    for epoch in range(100):
        loss = train(...)
        logger.log_scalar("train/loss", loss, step=epoch)
```

---

## 2. TensorBoard

### 2.1 Setup & Khởi chạy

```bash
# Activate environment
conda activate ain501

# Chạy TensorBoard
tensorboard --logdir runs/
# → Mở trình duyệt: http://localhost:6006
```

### 2.2 Cấu trúc thư mục logs

```
runs/
├── exp-001/                    # run_name="exp-001"
│   └── events.out.tfevents.*
├── exp-002/
│   └── events.out.tfevents.*
└── yolo-baseline/
    └── events.out.tfevents.*
```

> **Quy ước đặt tên `run_name`**: `<model>-<mô-tả>-<số>`, ví dụ: `yolo-baseline-001`, `resnet-finetune-003`

### 2.3 Tag naming convention

Dùng prefix phân nhóm để dashboard tự động group:

```python
# Training metrics
logger.log_scalar("train/loss", loss, step=epoch)
logger.log_scalar("train/lr", current_lr, step=epoch)

# Validation metrics
logger.log_scalar("val/loss", val_loss, step=epoch)
logger.log_scalar("val/accuracy", val_acc, step=epoch)
logger.log_scalar("val/mAP50", map50, step=epoch)

# Per-class metrics
logger.log_scalars("val/class_ap", {
    "person": 0.85,
    "car": 0.92,
    "bike": 0.78,
}, step=epoch)

# Model diagnostics
logger.log_histogram("gradients/layer1", grads, step=epoch)
logger.log_histogram("weights/fc1", weights, step=epoch)

# System
logger.log_scalar("system/gpu_mem_mb", mem_usage, step=epoch)
logger.log_scalar("system/epoch_time_s", epoch_time, step=epoch)
```

### 2.4 Các tab quan trọng trong TensorBoard

| Tab | Nội dung | Khi nào xem |
|-----|----------|-------------|
| **Scalars** | Loss, accuracy, learning rate | Mỗi lần train |
| **Images** | Predictions, augmented samples | Debug model output |
| **Histograms** | Weight/gradient distributions | Debug vanishing/exploding gradients |
| **Graphs** | Model architecture | Verify model structure |
| **Text** | Hyperparameters, notes | Review config |

### 2.5 So sánh nhiều runs

```bash
# TensorBoard tự động hiển thị tất cả runs trong --logdir
tensorboard --logdir runs/

# Hoặc chỉ so sánh 2 runs cụ thể
tensorboard --logdir exp1:runs/exp-001,exp2:runs/exp-002
```

---

## 3. Weights & Biases (W&B)

### 3.1 Setup lần đầu

```bash
# Cài đặt (đã có trong dependencies)
pip install wandb

# Đăng nhập (chỉ cần 1 lần)
wandb login
# → Nhập API key từ https://wandb.ai/authorize
```

### 3.2 Sử dụng trong code

```python
with TrainingLogger(
    project="ain501-yolo",            # Tên project trên W&B
    run_name="yolo-baseline-001",     # Tên run
    backends=["wandb"],
    config={
        "model": "YOLOv8n",
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 100,
        "device": "xpu",
        "img_size": 640,
    },
    tags=["baseline", "yolov8", "arc-a770"],
) as logger:
    # ... training loop ...
    pass
```

### 3.3 W&B Dashboard features

| Feature | Mô tả | Cách dùng |
|---------|--------|-----------|
| **Runs Table** | Bảng tất cả runs, sort/filter theo metric | Tự động |
| **Charts** | Auto-generated charts từ logged metrics | Tự động |
| **Run Comparison** | So sánh side-by-side nhiều runs | Chọn runs → Compare |
| **Sweep** | Hyperparameter search tự động | Cần config sweep riêng |
| **Artifacts** | Lưu model, dataset versions | `logger.log_artifact(...)` |
| **Reports** | Tạo báo cáo từ runs | W&B UI → Create Report |
| **Alerts** | Thông báo khi metric đạt threshold | W&B Settings |

### 3.4 Environment variables (optional)

```bash
# Tắt W&B khi debug (không upload)
set WANDB_MODE=offline

# Hoặc disable hoàn toàn
set WANDB_DISABLED=true

# Chỉ định team/entity
set WANDB_ENTITY=your-team-name
```

### 3.5 Log Artifacts (model, data)

```python
# Sau khi train xong
torch.save(model.state_dict(), "models/best.pt")
logger.log_artifact("models/best.pt", type="model")

# Log dataset version
logger.log_artifact("data/train/", type="dataset")
```

---

## 4. Metrics BẮT BUỘC phải log

### 4.1 Cho mọi training run

| Metric | Tag | Frequency |
|--------|-----|-----------|
| Training loss | `train/loss` | Mỗi epoch |
| Validation loss | `val/loss` | Mỗi epoch |
| Learning rate | `train/lr` | Mỗi epoch |
| Epoch time | `system/epoch_time_s` | Mỗi epoch |

### 4.2 Cho classification

| Metric | Tag |
|--------|-----|
| Accuracy | `val/accuracy` |
| F1 Score | `val/f1` |
| Precision | `val/precision` |
| Recall | `val/recall` |

### 4.3 Cho object detection (YOLO)

| Metric | Tag |
|--------|-----|
| mAP@50 | `val/mAP50` |
| mAP@50:95 | `val/mAP50-95` |
| Box loss | `train/box_loss` |
| Class loss | `train/cls_loss` |
| DFL loss | `train/dfl_loss` |

### 4.4 Cho OCR

| Metric | Tag |
|--------|-----|
| Character accuracy | `val/char_acc` |
| Word accuracy | `val/word_acc` |
| CER | `val/cer` |
| WER | `val/wer` |

---

## 5. Ví dụ hoàn chỉnh — YOLOv8 Training với Dashboard

```python
import torch
from ultralytics import YOLO
from src.utils import TrainingLogger

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Config
config = {
    "model": "yolov8n.pt",
    "data": "configs/dataset.yaml",
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "lr0": 0.01,
    "device": str(device),
}

with TrainingLogger(
    project="ain501-yolo",
    run_name="yolov8n-custom-001",
    backends=["tensorboard", "wandb"],
    log_dir="runs/detect",
    config=config,
    tags=["yolov8n", "custom-dataset"],
) as logger:

    model = YOLO(config["model"])

    # Ultralytics có built-in TensorBoard/W&B support,
    # nhưng dùng TrainingLogger để thống nhất logging pattern
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch_size"],
        imgsz=config["img_size"],
        device=str(device),
    )

    # Log final metrics
    logger.log_scalar("final/mAP50", results.results_dict["metrics/mAP50(B)"], step=config["epochs"])
    logger.log_artifact("runs/detect/yolov8n-custom-001/weights/best.pt", type="model")
```

---

## 6. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| TensorBoard không hiện data | Kiểm tra `--logdir` path chính xác |
| W&B không upload | Kiểm tra `wandb login`, network, `WANDB_MODE` |
| Quá nhiều data points | Tăng logging interval (log mỗi N steps) |
| Dashboard chậm | Giảm frequency `log_histogram`, `log_image` |
| W&B rate limit | Dùng `wandb.Settings(silent=True)`, batch logs |
| Port 6006 bị chiếm | `tensorboard --logdir runs/ --port 6007` |

---

## 7. Tóm tắt quy định

> [!IMPORTANT]
> 1. **Mọi training run phải có `TrainingLogger`** — không ngoại lệ.
> 2. **Metrics bắt buộc**: `train/loss`, `val/loss`, `train/lr`, `system/epoch_time_s`.
> 3. **Run name có ý nghĩa**: `<model>-<mô-tả>-<số>`.
> 4. **Tag naming**: dùng prefix group (`train/`, `val/`, `system/`).
> 5. **Train chính thức**: bật cả `tensorboard` + `wandb`.
> 6. **Lưu model**: luôn `log_artifact` model weights sau train.
