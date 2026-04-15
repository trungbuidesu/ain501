# Deploy Manifest (Runtime)

Runtime mục tiêu: clone repo + tải 1 file `models_release.zip` + chạy `python -m src.app` trong vài phút.

## 1) Runtime model artifacts

| File | Được load bởi | Bắt buộc/Optional | Size |
| --- | --- | --- | ---: |
| `models/yolov8n_int8.onnx` | `src/agents/object_agent.py` (qua `AppRuntime`/`ObjectAgent`) | Bắt buộc | 3.33 MB |
| `models/mobilenetv3_small_fp32.onnx` | `src/encoder/visual_encoder.py` (khi `encoder.enabled=true`) | Bắt buộc (theo config mặc định) | 3.54 MB |
| `models/minilm_l6.onnx` | `src/rag/minilm_embedder.py` (qua `KnowledgeBase`) | Bắt buộc (RAG bật mặc định) | 86.20 MB |
| `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx` | `src/output/audio_output.py` (`AudioOutputManager`) | Optional (có fallback `pyttsx3`) | 60.27 MB |
| `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json` | Piper voice metadata | Optional (đi cùng Piper ONNX) | 0.00 MB |
| `models/scene_classifier_int8.onnx` | `src/router/slm_router.py` (Tier 1) | Optional (Tier 1) | 1.75 MB |
| `models/movinet_a0_int8.onnx` | `src/agents/action_agent.py` (Tier 1) | Optional (Tier 1) | 4.42 MB |

Ghi chú:
- Không đóng gói training checkpoints (`*.pt`) và artifacts benchmark/training.
- Kích thước lấy từ máy hiện tại tại thời điểm audit.

## 2) Runtime data

| File | Vai trò | Bắt buộc |
| --- | --- | --- |
| `data/knowledge/hazard_patterns.json` | RAG knowledge | Có (RAG) |
| `data/knowledge/relationship_patterns.json` | RAG knowledge | Có (RAG) |
| `data/knowledge/spatial_templates.json` | RAG knowledge | Có (RAG) |

Không đóng gói data training:
- `data/raw/`
- `data/processed/`
- `data/custom/`

## 3) Runtime configs

| File | Vai trò | Bắt buộc |
| --- | --- | --- |
| `configs/app.yaml` | Config chính cho `python -m src.app` | Có |
| `configs/capture_pipeline.yaml` | Capture source/timing/change detection | Có |
| `configs/capture_webcam.yaml` | Preset capture webcam (optional preset) | Optional |

Không cần cho runtime app:
- `configs/datasets/*`
- phần lớn scripts training trong `configs/models/*` (không được app runtime đọc trực tiếp).

## 4) Runtime dependencies (curated)

`requirements.txt` được tạo riêng cho runtime, không dùng `pip freeze` full.

## 5) Packaging output target

- `models_release.zip` (chứa đúng runtime model files ở trên, giữ cấu trúc thư mục dưới `models/`).
- `scripts/verify_models.py` phải PASS trước khi chạy app.
