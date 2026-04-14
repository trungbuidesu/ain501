# Checklist MoViNet-A0 — Action Recognition

File này theo dõi trạng thái triển khai MoViNet-A0 cho nhận diện hành động real-time
trên Intel Arc A770 (XPU). Quy ước:

- `[x]` đã implement và có smoke/acceptance test pass.
- `[/]` đang thực hiện.
- `[ ]` chưa hoàn thành ở mức chạy thật trên domain data.
- `Trạng thái`: ghi rõ đang ở mức code-first, smoke test hay đã chạy domain data thật.

---

## 1. Hiểu Kiến Trúc MoViNet

- [x] Đọc paper: Mobile Video Networks (MoViNet)
  - Trạng thái: Đã nghiên cứu kiến trúc MoViNet-A0. Paper tập trung vào mobile
    video understanding với computational efficiency thông qua Neural Architecture
    Search (NAS) trên video classification tasks.

- [x] Hiểu stream buffer và causal convolutions
  - Trạng thái: MoViNet dùng `causal=True` để kích hoạt chế độ streaming: mỗi
    frame được xử lý độc lập thông qua activation buffer thay vì cần toàn bộ
    video. `TemporalCGAvgPool3D` thực hiện cumulative average theo thời gian.
    Đã implement `MoViNetA0Backbone(causal=True)` và verify streaming behavior
    trong `tests/test_movinet.py`.

- [x] Hiểu temporal modeling
  - Trạng thái: MoViNet-A0 dùng 3D depthwise convolutions kết hợp squeeze-excite
    temporal. Với `causal=True`, model chỉ nhìn vào past frames (không nhìn
    future) → phù hợp real-time inference.

- [x] Hiểu tại sao MoViNet phù hợp real-time hơn các model khác
  - Trạng thái: (1) Streaming buffer không cần buffer toàn bộ video; (2) ~2.6M
    params — nhỏ hơn S3D (~8M) và R3D (~33M); (3) Resolution gốc 172×172; (4)
    Latency thấp nhờ causal mode. Đã so sánh S3D vs MoViNet và chọn MoViNet.

---

## 2. Data Pipeline

- [x] Xây dựng VideoClipDataset (video → frames on-the-fly)
  - Trạng thái: Đã implement `src/training/data/action_recognition/clip_dataset.py`.
    Tuy nhiên phương pháp này ngốn CPU nặng khi train (decode video liên tục).

- [x] Migrate sang FrameClipDataset (đọc JPEG pre-decoded)
  - Trạng thái: Đã implement `src/training/data/action_recognition/frame_dataset.py`
    và script `preprocess_frames.py`. Đã chạy pre-decode toàn bộ 2,449 video →
    438,453 JPEG frames (7.3 GB) tại `data/processed/action_accessibility/frames/`.
    Manifest mới tại `data/processed/action_accessibility/manifest_frames.csv`.
    CPU usage khi train giảm từ ~100% xuống ~5–10%.

- [x] Pre-decode toàn bộ UCF-101 subset (2,449 videos)
  - Trạng thái: Đã chạy `python -m src.training.data.action_recognition.preprocess_frames
    --workers 6`. Hoàn thành 2449/2449, exit code 0. Dữ liệu split: 1,757 train /
    692 test.

- [x] Chọn 19 action classes accessibility
  - Trạng thái: Manifest đã có 19 classes từ UCF-101 subset. Xem file
    `configs/models/movinet_a0.yaml` để xem danh sách class.

---

## 3. Fine-tune MoViNet-A0

- [x] Load pretrained checkpoint Kinetics-600
  - Trạng thái: Đã implement load weights qua `movinets` library với
    `MoViNetA0Backbone(pretrained=True)`. Code tại `src/training/models/video/movinet.py`.

- [x] Modify classification head cho 19 classes mới
  - Trạng thái: Đã thay thế head phân loại (in_channels=2048 → 19 classes)
    bằng conv 1×1×1 cho cả `2plus1d` và `3d`. Verify trong test structure.

- [x] Config training hyperparameters
  - Trạng thái: `configs/models/movinet_a0.yaml` — lr=0.001, batch_size=32,
    epochs=50, n_clip_frames=6, frame_stride=2, resolution=172, temporal_jitter=True.

- [x] Chạy fine-tuning thực tế 50 epochs trên XPU
  - Trạng thái: **Đã hoàn thành** — Training 50 epochs trên Intel Arc A770. Loss giảm dần và Test Acc đạt khoảng 62.5% đến 73.1%. Các checkpoint (`best.pt`) đã được lưu thành công.

- [x] Log loss/accuracy lên TensorBoard mỗi epoch
  - Trạng thái: Đã có đủ logs tại `runs/movinet_a0_finetune`, quan sát được đường cong train/val metric trong suốt 50 epochs.

- [x] Lưu checkpoint tốt nhất
  - Trạng thái: Đã sinh model size ~10MB (Kích thước mô hình siêu nhẹ như kỳ vọng) tại `models/video/movinet_a0/best.pt`.

---

## 4. Evaluate

- [x] Top-1 accuracy trên test split
  - Trạng thái: Lệnh `evaluate` chạy với checkpoint xuất sắc cho ra **Top-1 Accuracy: 0.6257** (khoảng ~62.6% trên 19 classes với frame_stride=2 và random jitter tĩnh).

- [x] Confusion matrix
  - Trạng thái: Đã chạy qua script error analysis gộp.

- [x] Error analysis — actions hay bị nhầm lẫn
  - Trạng thái: Một số cặp hay nhầm lẫn nhất do đặc tính visual tương đồng:
    * `PullUps` -> `RopeClimbing` (chuyển động trục dọc tay)
    * `WallPushups` -> `BodyWeightSquats` (chuyển động dập lên/xuống toàn thân)
    * CLI `error-analysis` hoạt động trơn tru.

---

## 5. QAT (Quantization Aware Training)

- [ ] Prepare QAT
  - Trạng thái: Framework đã implement trong `main_qat()`. Cần checkpoint FP32
    từ bước fine-tune. Lệnh: `python -m src.training.scripts.movinet qat`

- [ ] Train thêm 10 epochs với QAT
  - Trạng thái: Chưa thực hiện — chờ FP32 checkpoint.

- [ ] Convert INT8
  - Trạng thái: Chưa thực hiện.

- [ ] So sánh accuracy FP32 vs INT8
  - Trạng thái: Chưa thực hiện.

---

## 6. Export ONNX

- [ ] Export ONNX FP32
  - Trạng thái: CLI `export-onnx` đã implement (opset_version=12). Hiện đang có
    blocker: `DispatchError` với `TemporalCGAvgPool3D` trong ONNX tracing. Cần
    investigate thêm sau khi có checkpoint thật.

- [ ] Verify ONNX graph
  - Trạng thái: Chưa thực hiện.

- [ ] Benchmark latency ONNX vs PyTorch
  - Trạng thái: Chưa thực hiện.

- [ ] Lưu `models/movinet_a0_int8.onnx`
  - Trạng thái: Chưa thực hiện.

---

## 7. CLI và Config

- [x] CLI thống nhất
  - Trạng thái: `python -m src.training.scripts.movinet describe|fine-tune|evaluate|
    error-analysis|qat|export-onnx` với `--config`, `--manifest`, `--frames-manifest`.

- [x] Config model defaults
  - Trạng thái: `configs/models/movinet_a0.yaml` — đầy đủ model, training, output params.

- [x] Không commit artifact nặng
  - Trạng thái: `.gitignore` đã ignore `models/video/`, `runs/`, `*.pt`, `*.onnx`,
    `data/processed/action_accessibility/frames/`.

---

## 8. Test / Acceptance

- [x] Unit tests backbone và streaming
  - Trạng thái: `tests/test_movinet.py` cover: backbone structure (4 layer groups),
    causal streaming (clean_activation_buffers), XPU compatibility (float32 dtype).
    **4/4 passed**.

- [x] Unit tests FrameClipDataset
  - Trạng thái: `test_clip_dataset` verify shape `[C, T, H, W]` và dtype uint8.
    Pass với manifest_frames.csv thực.

- [x] Quality gate
  - Trạng thái: `black`, `ruff check`, `mypy`, `pytest` — **45/45 passed**, 0 errors.
    Notebooks được exempt khỏi một số rule (B905, E402, F401) qua
    `extend-per-file-ignores` trong `pyproject.toml`.

---

## Ghi chú kỹ thuật

| Mục | Quyết định | Lý do |
|-----|-----------|-------|
| Resolution | 172×172 | Native resolution của MoViNet-A0 |
| Dataset format | JPEG frames (pre-decoded) | Giảm CPU bottleneck, GPU utilization ~90% |
| XPU dtype | float32 (ép tường minh) | Tránh `XPUDoubleType` error trên Arc A770 |
| ONNX opset | 12 | Ổn định hơn với MoViNet operators |
| Causal mode | True | Hỗ trợ real-time streaming inference |
| Batch size | 32 | Phù hợp 16GB VRAM Arc A770 |
| DataLoader workers | 4 (train), 2 (val) | JPEG nhẹ, 4 worker là dư sức |
