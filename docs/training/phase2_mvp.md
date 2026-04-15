# Phase 2 MVP: Object-to-Speech — Manual chi tiết

Tài liệu này mô tả cách cài đặt, cấu hình và chạy MVP Phase 2: **capture** → **change gate** → **YOLO detect** → **spatial template** → **Piper TTS**.

## 1. Pipeline tóm tắt

1. Đọc khung hình từ nguồn trong `configs/capture_pipeline.yaml` (webcam / màn hình / file video).
2. `ChangeDetector` bỏ qua khung gần như không đổi để giảm tải.
3. `ObjectAgent` chạy YOLOv8n ONNX INT8, NMS, trả danh sách `{label, confidence, bbox, count}`.
4. `SpatialTemplateEngine` sinh câu tiếng Việt (hướng + khoảng cách), ưu tiên hazard.
5. `AudioOutputManager` xếp hàng TTS Piper (ưu tiên / dedup), ghi WAV dưới `reports/app/audio/` (theo `output.reports_dir`).
6. Báo cáo độ trễ ghi vào `reports/app/app_latency.json`.

Mã entrypoint: `src/app/main.py` (`python -m src.app`).

### 1.1 Tier 1 (router + parallel agents)

Khi bật `tier1.enabled: true` trong `configs/app.yaml`, pipeline thêm **SLM router** (scene ONNX), **AgentManager** song song, buffer khung cho MoViNet, và các expert OCR / face-pose. Chi tiết kiến trúc, benchmark và troubleshooting: [phase3_tier1.md](phase3_tier1.md).

### 1.2 RAG Lite + orchestrator (Phase 3)

Caption có thể đi qua **MiniLM + FAISS** trên `data/knowledge/` khi `rag.enabled: true` trong [`configs/app.yaml`](../../configs/app.yaml). Fallback vẫn là spatial template engine. Chi tiết: [phase3_rag.md](phase3_rag.md).

## 2. Điều kiện cần

- Python 3.10+, env đã cài project (`pip install -e ".[dev]"`).
- **ONNX models** (đường dẫn mặc định trong `configs/app.yaml`):
  - `models/yolov8n_int8.onnx`
  - `models/mobilenetv3_small_fp32.onnx` (chỉ khi `encoder.enabled: true`)
- **Piper TTS**:
  - Gói `piper-tts` (extra `[tts]` trong `pyproject.toml`: `pip install -e ".[tts]"`).
  - File giọng `.onnx` và `.onnx.json` khớp `models.piper_model_path` trong config (xem mục 4).

Lưu ý: graph YOLO trong registry có thể dùng **batch tĩnh 16**; code đã tự lặp tensor đầu vào cho khớp batch đó.

## 3. Cài Piper và giọng (voice)

### 3.1 Cài package

```bash
pip install -e ".[tts]"
```

Trên Windows, executable thường nằm tại  
`<conda_env>\Scripts\piper.exe`. App sẽ tự thử resolve từ env hiện tại nếu lệnh `piper` không có trong `PATH`.

### 3.2 Tải voice (ví dụ tiếng Việt theo layout repo)

Đặt đúng cấu trúc thư mục như trong `configs/datasets/tts_piper_accessibility.yaml`, ví dụ:

- `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx`
- `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json`

Có thể tải từ Hugging Face `rhasspy/piper-voices` (chỉ các file cần cho voice đã chọn).

### 3.3 Kiểm tra nhanh Piper (tùy chọn)

```powershell
echo "xin chao" | & "$env:CONDA_PREFIX\Scripts\piper.exe" --model "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx" --output_file "reports/app/audio/manual_check.wav"
```

Chạy từ thư mục gốc repo; file WAV phải có kích thước > 0.

## 4. File cấu hình

### 4.1 `configs/app.yaml`

| Khóa | Ý nghĩa |
| --- | --- |
| `capture_config` | Đường tới YAML capture Phase 1 (mặc định `configs/capture_pipeline.yaml`). |
| `models.yolo_int8` | Đường ONNX YOLO INT8. |
| `models.mobilenet_fp32` | ONNX MobileNetV3 embedding (khi bật encoder). |
| `models.piper_model_path` | File `.onnx` giọng Piper. |
| `encoder.enabled` | `true`: chạy visual encoder mỗi khung đã emit; `false`: bỏ qua. |
| `detector.conf_threshold` | Ngưỡng confidence sau decode. |
| `detector.iou_threshold` | IoU cho NMS class-aware. |
| `detector.max_detections` | Giới hạn số box sau NMS. |
| `caption.empty_policy` | `silent`: không đọc khi không có object; hoặc dùng message (xem code/template). |
| `output.reports_dir` | Thư mục gốc báo cáo (mặc định `reports/app`). |
| `output.dedup_window_s` | Cửa sổ giây để không lặp lại cùng một “chữ ký” scene. |
| `output.debug_overlay` | `true`: cửa sổ PySide6 hiển thị caption gần nhất (dev). |
| `tier1.*` | Tier 1: `enabled`, `strict_artifacts`, `router`, `timeouts`, `ring_buffer_maxlen`, … — xem [phase3_tier1.md](phase3_tier1.md). |

### 4.2 `configs/capture_pipeline.yaml` (tham chiếu bởi `capture_config`)

| Khóa | Ý nghĩa |
| --- | --- |
| `source.type` | `webcam` \| `screen` \| `file`. |
| `source.webcam_index` | Chỉ số camera khi `webcam`. |
| `source.file_path` / `file_loop` | Video file khi `type: file`. |
| `screen.backend` | `auto` (ưu tiên dxcam rồi mss trên Windows). |
| `screen.region` | `null` hoặc `{left, top, width, height}` (pixel, virtual desktop). |
| `screen.select_region_on_start` | `true`: mở overlay kéo vùng một lần (chỉ `screen`). |
| `timing.target_fps` | FPS mục tiêu sau mỗi khung (sleep pacing). |
| `timing.buffer_size` | Kích thước ring buffer (Phase 2 vẫn dùng capture config). |
| `change_detection.*` | SSIM hoặc pixel diff; chỉ pipeline khi có thay đổi đủ lớn. |

Gợi ý: để test webcam, đổi `source.type: webcam` và tắt chọn vùng màn hình nếu không cần.

## 5. Lệnh chạy

Luôn chạy từ **thư mục gốc repo** (nơi có `configs/`).

### Bash / Git Bash

```bash
python -m src.app
```

### PowerShell (khi `python` không trong PATH)

```powershell
& "$env:CONDA_PREFIX\python.exe" -m src.app
```

### Smoke test (giới hạn số khung)

```bash
python -m src.app --max-frames 120
```

### Dừng app

Trong terminal đang chạy: `Ctrl + C`.

## 6. Bạn sẽ “thấy gì” khi chạy?

- **Không bật `debug_overlay`**: app chủ yếu chạy nền; phản hồi là **âm thanh** (Piper) và file WAV trong `reports/app/audio/`, cộng log JSON độ trễ.
- **`screen` + `select_region_on_start: true`**: một lần mở overlay kéo vùng màn hình (PySide6), sau đó capture theo vùng đó.
- **`output.debug_overlay: true`**: thêm cửa sổ nhỏ hiển thị caption mới nhất (tiện dev).

## 7. Đầu ra

- `reports/app/app_latency.json`: thống kê `processed_frames`, `emitted_frames`, độ trễ detect/template/e2e (ms).
- `reports/app/audio/tts_<timestamp>.wav`: từng lần synthesize (tên theo thời gian).

## 8. Xử lý sự cố thường gặp

| Hiện tượng | Hướng xử lý |
| --- | --- |
| Không có âm thanh | Kiểm tra `piper-tts` đã cài; file `.onnx` và `.onnx.json` đúng path; thử lệnh kiểm tra mục 3.3. |
| `piper` không tìm thấy trong PATH | Dùng env conda và/hoặc đảm bảo `Scripts\piper.exe` tồn tại; app có fallback resolve từ `sys.executable`. |
| Không phát hiện / ít emit | Giảm ngưỡng change detection hoặc chỉnh `ssim_trigger_below`; hoặc tăng chuyển động trước camera. |
| Chỉ thấy overlay chọn vùng rồi “im lặng” | Bình thường nếu scene ít đổi; hoặc bật `debug_overlay` để xem caption. |
| File WAV 0 byte | Thường do dừng app (`Ctrl+C`) khi Piper đang ghi; chạy lại hoặc tăng `--max-frames` để một lần synth kịp hoàn tất. |

## 9. Kiểm thử tự động (dev)

```bash
pytest tests/test_phase2_encoder.py tests/test_phase2_object_agent.py tests/test_phase2_template_engine.py tests/test_phase2_audio_output.py tests/test_main.py
```

## 10. Tài liệu liên quan

- Capture Phase 1: `src/capture/app.py`, `configs/capture_pipeline.yaml`
- Piper dataset config mẫu: `configs/datasets/tts_piper_accessibility.yaml`
- Registry model: `models/registry.json`
