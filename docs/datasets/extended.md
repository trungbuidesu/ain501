# Bộ Dữ Liệu Mở Rộng

Trang này gom nhóm các dataset không thuộc core hoặc đang ở trạng thái dự phòng.

## Vai Trò Trong Lộ Trình

- Các tác vụ trong trang này giúp mở rộng năng lực hệ thống, nhưng không chặn mốc core Phase 0.
- Mục tiêu chính là chuẩn bị sẵn config, IO contract và utility để kích hoạt khi cần.

## OCR Scene Text (Nhận diện chữ trong cảnh)

- Config: `configs/datasets/ocr_textocr.yaml`
- Nguồn: TextOCR 0.1
- Mục tiêu: nhận dạng chữ trong bối cảnh thực tế
- Input:
  - `data/external/textocr/train_val_images`
  - `data/external/textocr/TextOCR_0.1_train.json`
  - `data/external/textocr/TextOCR_0.1_val.json`
- Output:
  - `data/processed/ocr_textocr/annotations.jsonl`
- Lưu ý format:
  - Text field: `utf8_string`
  - BBox: `xywh_pixels`
  - Polygon field: `points`
  - Bỏ qua annotation `empty_text` hoặc `illegible`

### Nguồn Tải OCR

- Trang dữ liệu: [https://textvqa.org/textocr/](https://textvqa.org/textocr/)
- URL theo config:
  - `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip`
  - `https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json`
  - `https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json`

### Thông Số OCR

| Thông số | Giá trị |
| --- | --- |
| `task` | `ocr_scene_text` |
| `phase` | `0.7` |
| `annotation_version` | `0.1` |
| `source_format` | `textocr_json` |
| `normalized_output` | `data/processed/ocr_textocr/annotations.jsonl` |

## Video Captioning (Chú thích video)

- Config: `configs/datasets/video_captioning_msvd.yaml`
- Nguồn: MSVD
- Trạng thái: Tier 2 deferred
- Mục tiêu: ánh xạ `video_id` sang tập caption ứng viên
- Input:
  - `data/external/msvd/videos`
  - `data/external/msvd/annotations/msvd_captions.json`
- Output:
  - `data/processed/video_captioning_msvd/captions.json`
- Mapping chuẩn:
  - `video_id -> captions[]`

### Nguồn Tải Captioning

- Dataset mirror: [https://huggingface.co/datasets/friedrichor/MSVD](https://huggingface.co/datasets/friedrichor/MSVD)

### Thông Số Captioning

| Thông số | Giá trị |
| --- | --- |
| `task` | `video_captioning` |
| `phase` | `0.6` |
| `tier` | `2` |
| `normalized_mapping` | `video_id -> captions[]` |
| `normalized_output` | `data/processed/video_captioning_msvd/captions.json` |

## TTS Validation (Kiểm tra chuyển văn bản thành giọng nói)

- Config: `configs/datasets/tts_piper_accessibility.yaml`
- Nguồn: voice Piper local dưới `models/voices/piper`
- Mục tiêu: kiểm tra chất lượng output speech và runtime setup
- Voices mặc định:
  - `vi_VN-vais1000-medium`
  - `en_US-lessac-medium`
- Fallback voices:
  - `vi_VN-vivos-x_low`
  - `en_US-amy-medium`
- Kiểm tra chính:
  - đọc được config JSON,
  - WAV đọc được và có duration dương,
  - mono PCM,
  - sample rate khớp config.

### Nguồn Tải TTS

- Piper engine: [https://github.com/OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
- Piper voices: [https://huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

### Thông Số TTS

| Thông số | Giá trị |
| --- | --- |
| `task` | `tts_validation` |
| `phase` | `0.8` |
| `install_extra` | `.[tts]` |
| `voices_root` | `models/voices/piper` |
| `validation_output` | `data/tts_validation` |
| `expected_sample_rate_hz` | `22050` (mặc định vi/en) |

```bash
python -m src.training.scripts.data_utils validate-tts --dry-run
```

## Depth Estimation (Ước lượng độ sâu, dự phòng)

- Config: `configs/datasets/depth_midas_reserved.yaml`
- Trạng thái: reserved, không chặn mốc core hiện tại
- Mục tiêu: hỗ trợ monocular depth trong các phase sau
- Model family dự kiến: MiDaS/DPT
- IO contract:
  - Input: RGB image
  - Output: relative depth map
  - Không cam kết metric distance ở phase hiện tại.

### Nguồn Tải Depth

- Repository: [https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)

### Thông Số Depth

| Thông số | Giá trị |
| --- | --- |
| `task` | `monocular_depth_estimation` |
| `phase` | `0.9` |
| `status` | `reserved` |
| `model_root` | `models/depth/midas` |
| `input_root` | `data/processed/depth/input_frames` |
| `output_root` | `data/processed/depth/predictions` |

## Ghi Chú Chung

- Giữ dry-run làm mặc định khi có hỗ trợ.
- Chỉ dùng `--execute` sau khi đã check path local và yêu cầu riêng tư.
- Không commit artifact dữ liệu lớn vào Git.

## Tiêu Chí Sẵn Sàng Cho Mỗi Nhóm

- OCR: tạo được normalized annotations với schema đúng.
- Captioning: normalize được về `video_id -> captions[]`.
- TTS: pass toàn bộ validation checks của voice mặc định hoặc fallback.
- Depth: giữ config và path contract nhất quán cho integration phase sau.

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
