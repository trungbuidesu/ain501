# Bộ Dữ Liệu Mở Rộng

Trang này gom nhóm các dataset không thuộc core hoặc đang ở trạng thái dự phòng.

## OCR Scene Text (Nhận diện chữ trong cảnh)

- Config: `configs/datasets/ocr_textocr.yaml`
- Nguồn: TextOCR 0.1
- Mục tiêu: nhận dạng chữ trong bối cảnh thực tế

## Video Captioning (Chú thích video)

- Config: `configs/datasets/video_captioning_msvd.yaml`
- Nguồn: MSVD
- Trạng thái: Tier 2 deferred
- Mục tiêu: ánh xạ `video_id` sang tập caption ứng viên

## TTS Validation (Kiểm tra chuyển văn bản thành giọng nói)

- Config: `configs/datasets/tts_piper_accessibility.yaml`
- Nguồn: voice Piper local dưới `models/voices/piper`
- Mục tiêu: kiểm tra chất lượng output speech và runtime setup

```bash
python -m src.training.scripts.data_utils validate-tts --dry-run
```

## Depth Estimation (Ước lượng độ sâu, dự phòng)

- Config: `configs/datasets/depth_midas_reserved.yaml`
- Trạng thái: reserved, không chặn mốc core hiện tại
- Mục tiêu: hỗ trợ monocular depth trong các phase sau

## Ghi Chú Chung

- Giữ dry-run làm mặc định khi có hỗ trợ.
- Chỉ dùng `--execute` sau khi đã check path local và yêu cầu riêng tư.
- Không commit artifact dữ liệu lớn vào Git.

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
