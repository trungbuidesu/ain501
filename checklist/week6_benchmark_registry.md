# Week 6 — benchmark registry and handoff

Convention: `[x]` done, `[/]` in progress, `[ ]` not started.

## Trong repo (đã có sẵn)

- [x] `models/registry.json` (schema v1): inventory ONNX FP32/INT8 (hoặc `null`) + `metrics_json` + `input_specs` cho YOLOv8n, MoViNet-A0, scene classifier, MobileNetV3-Small, MiniLM
- [x] `configs/benchmark/week6.yaml` + `src/training/scripts/benchmark_suite.py` (bảng MD/JSON, rubric)
- [x] `src/training/scripts/verify_onnx_registry.py` (một forward ORT; exit ≠ 0 nếu load/infer lỗi)
- [x] Active learning MVP: `src/training/scripts/active_learning_capture.py`, `active_learning_merge.py`, `docs/active_learning.md`
- [x] `docs/training/week6_training_log.md`; liên kết Tuần 6 trong `docs/models/index.md` và `mkdocs.yml`
- [x] Smoke tests: `tests/test_week6_registry.py`

## Trên máy (nghiệm thu) — đã chạy với Miniconda

Môi trường: `C:\Users\bdtrung29.1\miniconda3`, env **`ain501`**:  
`C:\Users\bdtrung29.1\miniconda3\envs\ain501\python.exe`

- [x] **Metric JSON**: `eval_summary.json` từ `yolov8n evaluate` (weights mặc định); `mobilenetv3_small/latest_summary.json` từ fine-tune `week6_real_smoke`; `scene_classifier/latest_summary.json` có `test_accuracy: null` cho đến khi có manifest + train.
- [x] **ONNX MobileNet**: `models/mobilenetv3_small_fp32.onnx` export từ `best_backbone.pt` của run `week6_real_smoke`.
- [x] Benchmark đầy đủ (ORT): `python -m src.training.scripts.benchmark_suite --config configs/benchmark/week6.yaml` → `reports/week6/benchmark_report.{md,json}`
- [x] Verify ONNX: **OK** toàn bộ entry trong registry (gồm `mobilenetv3_small:fp32_onnx`).
- [x] Báo cáo Tuần 6 đủ **5 dòng model**; cột metric đã điền trừ chỗ INT8 không có (MiniLM, v.v.).
- [x] Rubric: xem `reports/week6/benchmark_report.md` → mục *Selection rubric*; ghi thêm quyết định tay trong `docs/training/week6_training_log.md` nếu khác gợi ý tự động.
- [x] Đã đọc / cập nhật nội dung `docs/training/week6_training_log.md` và `docs/active_learning.md` (đường dẫn Python + mục metric thiếu).

## Acceptance

- Registry mô tả đúng **đường dẫn định** tới artifact (file nặng / ONNX thường gitignore — clone hoặc train để có file).
- Verify ONNX **pass** cho mọi artifact **có mặt** và hợp lệ; báo cáo benchmark đã tạo trên máy dev (Windows + ORT CPU).
- `reports/` trong `.gitignore` — giữ bản local `reports/week6/*` khi cần nộp bài.
