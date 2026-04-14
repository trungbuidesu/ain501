# Checklist YOLOv8n - Detection, Fine-Tune, QAT

File này theo dõi trạng thái YOLOv8n cho object detection. Quy ước:

- `[x]` đã có trong repo hoặc đã được chuẩn bị ở mức dependency/config/helper.
- `[ ]` chưa implement hoặc chưa chạy thật.
- `Trạng thái`: ghi rõ đang ở mức code-first, dataset helper, hay cần training/evaluation thật.

## YOLOv8n Architecture Understanding

- [x] Backbone: CSPDarknet / C2f-style feature extractor
  - Trạng thái: CLI `python -m src.training.scripts.yolov8n describe` đã in summary backbone CSPDarknet-style/C2f/SPPF cho YOLOv8n.
- [x] Neck: PANet/FPN-style feature fusion
  - Trạng thái: CLI `describe` đã ghi PAN-FPN top-down/bottom-up multi-scale feature fusion.
- [x] Head: decoupled detection head
  - Trạng thái: CLI `describe` đã ghi decoupled anchor-free detection head với box/class/DFL outputs.
- [x] Anchor-free detection
  - Trạng thái: CLI `describe` đã report `anchor_free=true`.
- [x] Loss composition: box + cls + dfl
  - Trạng thái: CLI `describe` đã report loss components `box`, `cls`, `dfl`.
- [x] NMS postprocessing
  - Trạng thái: CLI `describe` đã ghi confidence filtering + NMS; `predict-errors` đọc output prediction sau postprocess ở dry-run-safe wrapper.

## Fine-Tune YOLOv8n

- [x] Dependency Ultralytics
  - Trạng thái: `pyproject.toml` đã khai báo `ultralytics>=8.0.0`; env `trungbd` hiện detect được Ultralytics.
- [x] Dataset class config
  - Trạng thái: `configs/datasets/object_detection_accessibility.yaml` đã có `classes.coco_subset`, `classes.custom_only`, `output.yolo_root`, `output.merged_root`, và `canonical_names_path`.
- [x] COCO/custom -> YOLO dataset helpers
  - Trạng thái: Đã có `convert`, `validate`, `merge-yolo`, `normalize-custom-yolo` trong `src.training.scripts.data_utils`; dùng để chuẩn bị layout `images/<split>` + `labels/<split>`.
- [x] Training config YAML cho YOLOv8n
  - Trạng thái: Đã có `configs/models/yolov8n.yaml` và CLI `write-data-yaml` sinh Ultralytics data YAML từ canonical YOLO dataset/classes.
- [x] Classes + augmentations + hyperparams trong training config
  - Trạng thái: Config đã khai báo `lr0=0.01`, `epochs=100`, `batch=16`, `imgsz=640`, `mosaic=1.0`, `mixup=0.1`, `multi_scale=true`.
- [ ] Train trên COCO hoặc custom dataset
  - Trạng thái: Đã có CLI `train` dry-run mặc định và `--execute` mới gọi Ultralytics; chưa chạy real long training/mAP trong batch này.
- [x] Base LR `0.01`, epochs `50-100`
  - Trạng thái: Đã cấu hình `lr0=0.01`, `epochs=100` trong `configs/models/yolov8n.yaml` và expose override qua CLI.
- [x] Mosaic + mixup augmentation
  - Trạng thái: Đã cấu hình `mosaic=1.0`, `mixup=0.1`; CLI `train|ablate` cho phép override.
- [x] Multi-scale training
  - Trạng thái: Đã cấu hình `multi_scale=true`; CLI hỗ trợ `--multi-scale/--no-multi-scale`.

## Monitor / Evaluate / Error Analysis

- [ ] Monitor loss curves
  - Trạng thái: CLI `train` đã sẵn sàng gọi Ultralytics khi `--execute`; chưa tạo real run artifact/loss curves trong batch này.
- [ ] Monitor `mAP@0.5`
  - Trạng thái: CLI `evaluate` đã dry-run-safe; chưa chạy real checkpoint evaluation.
- [ ] Monitor `mAP@0.5:0.95`
  - Trạng thái: CLI `evaluate` đã dry-run-safe; chưa chạy real checkpoint evaluation.
- [ ] Evaluate trên val set -> report mAP
  - Trạng thái: CLI `evaluate` đã có; real mAP report vẫn mở vì chưa chạy long training/evaluation.
- [x] Error analysis: xem predictions sai, tìm pattern
  - Trạng thái: Đã có CLI `predict-errors` dry-run mặc định; với `--execute` sẽ predict trên sample val và xuất CSV image path, classes, confidences, artifact path.
- [x] Optional ablation study: thử bỏ/thêm augmentation, thay LR
  - Trạng thái: Đã có CLI `ablate` tạo matrix baseline/no_mosaic/no_mixup/lower_lr và chỉ chạy thật khi thêm `--execute`.

## QAT YOLOv8n

- [x] Level 1: dùng `torch.quantization.prepare_qat()` trên model
  - Trạng thái: Đã có CLI `qat` guard report `supported=false`; không chạy fake QAT khi chưa chứng minh compatibility trực tiếp với Ultralytics YOLOv8n.
- [ ] Train thêm 10-20 epochs với fake quant
  - Trạng thái: Chưa có QAT training loop hoặc config epochs riêng.
- [ ] `torch.quantization.convert()` -> INT8 model
  - Trạng thái: Chưa có export/convert artifact INT8.
- [ ] So sánh mAP: FP32 vs INT8
  - Trạng thái: Chưa có report so sánh metric giữa baseline FP32 và INT8.

## Đề Xuất Batch Tiếp Theo

- [x] Thêm `configs/models/yolov8n.yaml`
  - Trạng thái: Đã thêm model name `yolov8n.pt`, dataset paths, `lr0=0.01`, `epochs=100`, `mosaic`, `mixup`, `multi_scale`, output dirs.
- [x] Thêm CLI `python -m src.training.scripts.yolov8n`
  - Trạng thái: Đã thêm subcommand `describe`, `write-data-yaml`, `train`, `evaluate`, `predict-errors`, `ablate`, `qat`; lệnh train/evaluate/predict/ablate dry-run mặc định.
- [x] Thêm tests synthetic/dry-run
  - Trạng thái: Đã thêm `tests/test_yolov8n.py` cover describe, write-data-yaml dry-run/execute, train/evaluate/predict-errors/ablate dry-run, và QAT guard.
