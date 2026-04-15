# Checklist Scene Classifier - Router

File này theo dõi trạng thái triển khai Scene Classifier. Quy ước:

- `[x]` đã có code và smoke test pass.
- `[ ]` chưa hoàn thành ở mức chạy thật trên domain data.
- `Trạng thái`: nêu rõ mức smoke hay production.

## Class and Architecture

- [x] Define classes: `static`, `motion`, `text_present`, `face_detected`, `scene_change`
  - Trạng thái: Đã cố định class order trong `configs/models/scene_classifier.yaml` và `SCENE_ROUTER_CLASSES`.
- [x] Tiny CNN (`3-4` conv + FC, ~1M params)
  - Trạng thái: Đã implement `TinySceneCNN` trong `src/training/models/vision/scene_classifier.py`.
- [x] MobileNet lightweight transfer head
  - Trạng thái: Đã implement `MobileNetSceneHead` + builder `build_scene_classifier`.

## Data Pipeline

- [x] Build unified manifest `path,label,source,split`
  - Trạng thái: Đã có `scene_router_dataset.py` và CLI `prepare-manifest`.
- [x] Source mapping theo 5 class (Places/UCF/TextOCR/Face/Scene change)
  - Trạng thái: Đã implement collector functions cho từng class và merge builder.
- [ ] Reach `500-1000` samples per class trên dữ liệu thật
  - Trạng thái: Pipeline hỗ trợ `--min-per-class`/`--max-per-class`; chưa chạy full domain dataset.
- [x] Balance policy + augmentation planning
  - Trạng thái: Đã có weighted sampling trong train loader và `augment_balance_plan`.

## Training and Evaluation

- [x] Train-from-scratch loop (PyTorch)
  - Trạng thái: Đã implement vòng train chuẩn `zero_grad -> forward -> loss -> backward -> step`.
- [x] Track loss, accuracy, per-class precision/recall/F1
  - Trạng thái: Đã log epoch metrics và per-class val precision/recall.
- [x] Confusion matrix + misclassification export
  - Trạng thái: Đã xuất `confusion_matrix.csv` và `misclassifications.csv` trong thư mục run.
- [ ] Achieve `>85%` accuracy trên test set thật
  - Trạng thái: Có gate `--target-accuracy` + `--enforce-target-accuracy`; chưa có run domain data chứng minh.

## QAT, ONNX, Benchmark

- [x] QAT training flow
  - Trạng thái: Đã có subcommand `qat` với backend fallback theo engine hỗ trợ.
- [x] Export ONNX INT8 + parity verification
  - Trạng thái: Đã có `export-onnx` tạo FP32/INT8 ONNX, parity top-1 report.
- [x] Save final model at `models/scene_classifier_int8.onnx`
  - Trạng thái: Default output path đã cấu hình trong `configs/models/scene_classifier.yaml`.
- [ ] Benchmark `< 5ms` on target CPU thật
  - Trạng thái: Đã có `benchmark` report `latency_ms_p50/p95`; chưa có số liệu production target.

## Tests and Docs

- [x] Add tests for model/data/train/QAT/export/benchmark smoke
  - Trạng thái: `tests/test_scene_classifier.py` pass trong env `ain501`.
- [x] Update docs
  - Trạng thái: Đã cập nhật `docs/models/scene_classifier.md` và `docs/models/index.md`.
