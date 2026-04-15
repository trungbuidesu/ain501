# Models

Thư mục này chứa ONNX models cho app runtime.
Model binaries KHÔNG được commit vào git do kích thước lớn.

## Download

- Tải `models_release.zip` từ link phát hành nội bộ (hoặc Google Drive của team).
- Giải nén vào thư mục `models/` tại root repo.
- Hoặc copy nguyên thư mục `models/` từ máy đã có sẵn models.

## Models cần thiết

| Model | File | Size | Bắt buộc |
| --- | --- | ---: | --- |
| YOLOv8n | `yolov8n_int8.onnx` | ~3.3 MB | Có |
| MobileNetV3 | `mobilenetv3_small_fp32.onnx` | ~3.5 MB | Có |
| MiniLM-L6 | `minilm_l6.onnx` | ~86 MB | Có (khi RAG bật) |
| Scene classifier | `scene_classifier_int8.onnx` | ~1.8 MB | Không (Tier 1) |
| MoViNet-A0 | `movinet_a0_int8.onnx` | ~4.4 MB | Không (Tier 1) |
| Piper TTS Vi | `tts/piper/.../vi_VN-vais1000-medium.onnx` + `.json` | ~60 MB | Không (fallback `pyttsx3`) |

## Verify

```bash
python scripts/verify_models.py
```
