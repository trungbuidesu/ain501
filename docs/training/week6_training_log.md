# Week 6 training and benchmark notes

Consolidated lessons while unifying ONNX benchmarks, the model registry, and deployment-oriented metrics.

## Hyperparameters (reference)

- **YOLOv8n**: See `configs/models/yolov8n.yaml` and pilot reports under `reports/yolov8n/`; pilot runs used fewer epochs than the full 100-epoch target—treat pilot mAP as indicative, not final.
- **MoViNet-A0**: HMDB fine-tune in `configs/models/movinet_a0.yaml`; top-1 in `reports/movinet_a0/hmdb_retrain_summary.json`.
- **Scene classifier**: `configs/models/scene_classifier.yaml`, manifest via `scene_classifier prepare-manifest`.
- **MobileNetV3-Small**: Embedding head settings in `configs/models/mobilenetv3_small.yaml`.
- **MiniLM-L6**: `configs/models/minilm_l6.yaml`; retrieval metric in `reports/minilm_l6/check_run_summary.json`. Export and ONNX checks may need `optimum` / compatible `transformers` versions.

## YOLOv8n: low mAP on `yolov8n_data.yaml` vs ~52 on COCO (sanity check)

**Verified (official COCO val):** `YOLO('yolov8n.pt').val(data='coco.yaml', imgsz=640)` on this machine produced **mAP50 ≈ 0.519** (~52%), **mAP50-95 ≈ 0.368**. The pretrained checkpoint is healthy; the gap is not from broken weights.

**Why accessibility merged val shows ~0.09 mAP with the same `yolov8n.pt`:**

1. **Class index space:** `yolov8n.pt` is trained for **COCO 80 classes** with fixed category indices (Ultralytics order). `yolov8n_data.yaml` uses **`nc: 16`** and a **project-specific order** (`classes.txt`: person, dog, bicycle, …). Local YOLO labels use ids **0…15** in that order. COCO’s id for **dog** is **16**, not **1**; **bicycle** is **1** in COCO but **2** in the subset list. During `val`, predictions stay in **COCO index space** while GT labels are in **subset 0…15**, so matches are wrong for most classes except where indices accidentally align (e.g. **person** is 0 in both). That collapses mean AP — this is **label mapping**, not `conf` or `imgsz`.

2. **Val format:** Images + YOLO txt under `images/val` and `labels/val` are standard; **imgsz=640** matches training defaults. **`model.val()`** does not use a “too high” conf for mAP; Ultralytics uses evaluation defaults (low conf threshold for recall on the PR curve).

3. **What to do for a fair accessibility metric:** Fine-tune so the head matches **16 classes**; **or** keep labels in **COCO original category ids** in the label files and use a **80-class** data yaml (with only the 16 names you care about for reporting); **or** evaluate with a script that maps predicted COCO class id → subset id before computing AP.

## Pitfalls

1. **Metric comparability**: Do not compare mAP with top-1 or recall@k in one scalar “accuracy”. The Week 6 table keeps **metric name + value** per task.
2. **Static batch ONNX (YOLO)**: Latency is normalized per image when `latency_normalization.divide_batch_by` is set in `models/registry.json`.
3. **INT8 absent**: Some models only ship FP32 ONNX; the report shows `N/A` for the missing precision and the rubric skips unfair comparisons.
4. **Integer inputs (text)**: MiniLM ONNX expects `int64` token tensors; dummy feeds must be integer arrays, not floats.
5. **Missing local binaries**: Registry paths point to local artifacts; fresh clones must train or copy weights before `verify_onnx_registry` can load every file.

## Benchmark commands

Windows (Miniconda env `ain501`): use `conda activate ain501`, or call the interpreter explicitly:

`C:\Users\bdtrung29.1\miniconda3\envs\ain501\python.exe`

```bash
python -m src.training.scripts.benchmark_suite --config configs/benchmark/week6.yaml
python -m src.training.scripts.benchmark_suite --config configs/benchmark/week6.yaml --skip-benchmark
python -m src.training.scripts.verify_onnx_registry
```

Outputs: `reports/week6/benchmark_report.md` and `.json`. Selection hints use `configs/benchmark/week6.yaml` → `rubric`.

### Tuần 6 — quyết định gợi ý (tự động)

Xem bảng **Selection rubric** trong `reports/week6/benchmark_report.md`. Ghi chú tay tại đây nếu bạn chọn khác (latency thực tế trên thiết bị đích, hạn mức bộ nhớ, v.v.).

### Metric summary files (Week 6) — trạng thái thực tế

- `reports/yolov8n/pilot_5e/eval_summary.json` — từ **`yolov8n evaluate --preset pilot --execute`** (weights mặc định `yolov8n.pt`, val trên `yolov8n_data.yaml`). Số **mAP50 / mAP50-95** là đo Ultralytics trên máy này; không phải checkpoint pilot fine-tune trừ khi bạn truyền `--weights` trỏ tới `best.pt` đã train.
- `reports/scene_classifier/latest_summary.json` — **`test_accuracy`: null** vì chưa chạy train: `prepare-manifest` lỗi thiếu lớp `face_detected` (cần dữ liệu face theo `scene_classifier prepare-manifest`). Sau khi train thành công, file này được ghi tự động cùng `summary.json` trong thư mục checkpoint.
- `reports/mobilenetv3_small/latest_summary.json` — **`best_val_accuracy` thật** từ fine-tune ngắn (`run_name=week6_real_smoke`, 2 epoch, manifest `mobilenetv3_small_scene`). ONNX FP32: export từ `best_backbone.pt` của run đó.

**Lệnh tham chiếu**

```text
python -m src.training.scripts.yolov8n --config configs/models/yolov8n.yaml evaluate --preset pilot --execute --device cpu --summary-json reports/yolov8n/pilot_5e/eval_summary.json
python -m src.training.scripts.mobilenetv3 --config configs/models/mobilenetv3_small.yaml fine-tune --manifest-csv data/processed/mobilenetv3_small_scene/manifest.csv --epochs 2 --device cpu --run-name week6_real_smoke
python -m src.training.scripts.mobilenetv3 --config configs/models/mobilenetv3_small.yaml export-onnx --checkpoint models/vision/mobilenetv3_small/week6_real_smoke/best_backbone.pt --output models/mobilenetv3_small_fp32.onnx --output-kind embedding
```

## Lessons learned

- Central `models/registry.json` reduces one-off path drift between scripts.
- One ORT code path (`src/utils/onnx_benchmark.py`) keeps RSS and latency methodology consistent with earlier per-model scripts.
- Active learning stays intentionally small: inbox + CSV + merge script, documented in `docs/active_learning.md`.

## Code review notes (current state)

This pass reviewed the new capture stack and YOLO remap evaluation path and then applied focused fixes.

### Findings and resolution

1. **High — YOLO remap script default device could fail on Intel Arc/XPU envs**
   - **Issue:** `eval_yolov8_accessibility_remap.py --device auto` could resolve to `xpu`, but current Ultralytics detection API in this environment rejected `xpu` for `.predict()`, causing immediate runtime failure.
   - **Fix applied:** `resolve_predict_device()` now detects `xpu*` and falls back to `cpu` with a warning.
   - **Impact:** Script is now robust in Arc environments without requiring manual device override.

2. **Medium — Capture loop `max_frames` off-by-one**
   - **Issue:** In `run_capture_app`, stop condition used `n_read >= max_frames` before processing, so one frame was skipped from processing when hitting the boundary.
   - **Fix applied:** Condition changed to `n_read > max_frames`, which preserves exactly `max_frames` processed frames.

3. **Medium — `screen.backend` config was parsed but not enforced**
   - **Issue:** `configs/capture_pipeline.yaml` exposes `screen.backend` (`auto|dxcam|mss`) but `frame_source_from_config()` always used default `ScreenSource(...)`, effectively ignoring backend choice.
   - **Fix applied:** `frame_source_from_config()` now passes `backend=cfg.screen.backend`. `ScreenSource` validates backend values, raises for unsupported combinations (`dxcam` + explicit region), and honors explicit backend selection.

### Test coverage updates added in this pass

- `tests/test_yolov8_coco_remap.py`
  - Added device resolution checks (`cpu` pass-through, `xpu` fallback).
- `tests/test_capture_phase1.py`
  - Added assertion that `screen.backend` from config is forwarded to source construction.
  - Added `run_capture_app` regression test for `max_frames` boundary behavior.

Targeted test run:

```text
pytest tests/test_yolov8_coco_remap.py tests/test_capture_phase1.py -q
# 22 passed, 2 skipped
```

## Checklist cross-links

- `checklist/yolov8n.md`
- `checklist/movinet_a0.md`
- `checklist/scene_classifier.md`
- `checklist/mobilenetv3_small.md`
- `checklist/minilm_l6.md`
- `checklist/week6_benchmark_registry.md`
