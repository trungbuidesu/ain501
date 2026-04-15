# Kiến Trúc Nhận Diện Số Hóa: PaddleOCR Lite

> [!NOTE] Bản Nháp Tài Liệu (Draft Note)
> Trang tài liệu này đang trong tiến trình soạn thảo (Work in Progress).

Trang dữ liệu quy chuẩn dành cho mạng OCR gọn nhẹ v2/v3 của khối PaddleOCR (~4M tham số, dung lượng xấp xỉ 8MB). Giới hạn thiết kế phục vụ mục đích trích xuất và giải mã lớp đồ họa văn tự tự nhiên và biển báo đường phố (Text / Sign reading). Chi tiết về cấu trúc CRNN sẽ được đăng tải sớm.

## Week 5 verification (CPU, no fine-tuning)

**Install:** `paddleocr` is listed in `pyproject.toml`. You still need a matching **PaddlePaddle** CPU wheel for your OS/Python (see the [official install guide](https://www.paddlepaddle.org.cn/install/quick)). Record `paddle` and `paddleocr` versions in `reports/prebuilt_week5/paddleocr_benchmark_cpu.json`.

**CLI:**

```text
python -m src.training.scripts.verify_prebuilt_week5 paddleocr --synthetic --limit 20
```

**Outputs:**

- `reports/prebuilt_week5/paddleocr_samples.jsonl` — one JSON object per image.
- `reports/prebuilt_week5/paddleocr_benchmark_cpu.json` — p50/p95 latency, warmup, repeat benchmark on first image.

**JSONL schema (one line per image):**

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | string | Path to input image |
| `source` | string | `synthetic` or `user_dir` |
| `lines` | array | OCR detections |
| `lines[].box` | array | Quadrilateral `[[x,y],...]` (pixel coords) |
| `lines[].text` | string | Recognized text |
| `lines[].score` | float | Confidence |
| `elapsed_ms` | float | Wall time for full OCR on that image |

**Example row:**

```json
{"image_id": "reports/prebuilt_week5/_synth_images/synth_00.png", "source": "synthetic", "lines": [{"box": [[0,0],[1,0],[1,1],[0,1]], "text": "HELLO 0", "score": 0.99}], "elapsed_ms": 120.5}
```