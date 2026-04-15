# App Runtime: Object-to-Speech — Manual chi tiết

Tài liệu này mô tả cách cài đặt, cấu hình và chạy app runtime:
**capture** -> **change gate** -> **YOLO detect** -> **spatial template/RAG** -> **Piper TTS**.

## 1. Pipeline tóm tắt

1. Đọc khung hình từ nguồn trong `configs/capture_pipeline.yaml` (webcam / màn hình / file video).
2. `ChangeDetector` bỏ qua khung gần như không đổi để giảm tải.
3. `ObjectAgent` chạy YOLOv8n ONNX INT8, NMS, trả danh sách `{label, confidence, bbox, count}`.
4. `SpatialTemplateEngine` sinh câu tiếng Việt (hướng + khoảng cách), ưu tiên hazard.
5. `AudioOutputManager` xếp hàng TTS Piper (ưu tiên / dedup), ghi WAV dưới `reports/app/audio/` (theo `output.reports_dir`).
6. Báo cáo độ trễ ghi vào `reports/app/app_latency.json`.

Entrypoint: `src/app/main.py` (`python -m src.app`).

### 1.1 Tier 1 (router + parallel agents)

Khi bật `tier1.enabled: true` trong `configs/app.yaml`, pipeline thêm **SLM router** (scene ONNX), **AgentManager** song song, buffer khung cho MoViNet, và các expert OCR / face-pose. Chi tiết kiến trúc, benchmark và troubleshooting: [tier1_pipeline.md](tier1_pipeline.md).

### 1.2 RAG Lite + orchestrator

Caption có thể đi qua **MiniLM + FAISS** trên `data/knowledge/` khi `rag.enabled: true` trong [`configs/app.yaml`](../../configs/app.yaml). Fallback vẫn là spatial template engine. Chi tiết: [rag_orchestrator.md](rag_orchestrator.md).

## 2. Điều kiện cần

- Python 3.10+, env đã cài project (`pip install -e ".[dev]"`).
- ONNX models (đường dẫn mặc định trong `configs/app.yaml`):
  - `models/yolov8n_int8.onnx`
  - `models/mobilenetv3_small_fp32.onnx` (khi `encoder.enabled: true`)
- Piper TTS:
  - Gói `piper-tts` (`pip install -e ".[tts]"`).
  - File giọng `.onnx` và `.onnx.json` khớp `models.piper_model_path`.

## 3. Cài Piper và giọng

```bash
pip install -e ".[tts]"
```

## 4. File cấu hình

### 4.1 `configs/app.yaml`

- `capture_config`: YAML capture (mặc định `configs/capture_pipeline.yaml`)
- `models.*`: model paths
- `detector.*`: ngưỡng detect
- `caption.empty_policy`: hành vi khi rỗng
- `output.*`: `reports_dir`, `dedup_window_s`, `debug_overlay`
- `tier1.*`: router/experts/timeouts
- `rag.*`: retriever/knowledge base

### 4.2 `configs/capture_pipeline.yaml`

Thiết lập nguồn `webcam|screen|file`, FPS, và `change_detection`.

## 5. Lệnh chạy

```bash
python -m src.app
python -m src.app --max-frames 120
python -m src.app --dry-run
python -m src.app --visual --dry-run
```

## 6. Đầu ra

- `reports/app/app_latency.json`
- `reports/app/audio/tts_<timestamp>.wav`

## 7. Kiểm thử tự động

```bash
pytest tests/test_visual_encoder.py tests/test_object_agent.py tests/test_spatial_template_engine.py tests/test_audio_output_manager.py tests/test_main.py
```

## 8. Tài liệu liên quan

- Capture: `src/capture/app.py`, `configs/capture_pipeline.yaml`
- Tier1: [tier1_pipeline.md](tier1_pipeline.md)
- RAG: [rag_orchestrator.md](rag_orchestrator.md)
