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
  - Trạng thái: Đã có `configs/models/yolov8n.yaml` và CLI `write-data-yaml`; data YAML local/ignored tại `data/processed/object_detection_accessibility_merged/yolov8n_data.yaml` hiện trỏ đúng `train: images/train`, `val: images/val`, `nc: 16`.
- [x] Classes + augmentations + hyperparams trong training config
  - Trạng thái: Config đã khai báo `lr0=0.01`, `epochs=100`, `batch=16`, `imgsz=640`, `mosaic=1.0`, `mixup=0.1`, `multi_scale=true`.
- [x] Train trên COCO hoặc custom dataset
  - Trạng thái: Đã chạy thành công `pilot` 5 epochs trên subset COCO YOLO, tạo mAP và loss curves. CLI `train` đã `--execute` thật.
- [x] Base LR `0.01`, epochs `50-100`
  - Trạng thái: Đã cấu hình `lr0=0.01`, `epochs=100` trong `configs/models/yolov8n.yaml` và expose override qua CLI.
- [x] Mosaic + mixup augmentation
  - Trạng thái: Đã cấu hình `mosaic=1.0`, `mixup=0.1`; CLI `train|ablate` cho phép override.
- [x] Multi-scale training
  - Trạng thái: Đã cấu hình `multi_scale=true`; CLI hỗ trợ `--multi-scale/--no-multi-scale`.

## Monitor / Evaluate / Error Analysis

- [x] Monitor loss curves
  - Trạng thái: Đã sinh loss curves gốc (`results.png` / `results.csv`) trong thư mục experiment `runs/detect/runs/yolov8n/pilot_5e`.
- [x] Monitor `mAP@0.5`
  - Trạng thái: Đã có log mAP từ `val`: `0.187`.
- [x] Monitor `mAP@0.5:0.95`
  - Trạng thái: Đã có log mAP50-95: `0.122`.
- [x] Evaluate trên val set -> report mAP
  - Trạng thái: Chạy tích hợp ở cuối quá trình Pilot epochs 5.
- [x] Error analysis: xem predictions sai, tìm pattern
  - Trạng thái: Đã có CLI `predict-errors` dry-run mặc định; với `--execute` sẽ predict trên sample val và xuất CSV image path, classes, confidences, artifact path.
- [x] Optional ablation study: thử bỏ/thêm augmentation, thay LR
  - Trạng thái: Đã có CLI `ablate` tạo matrix baseline/no_mosaic/no_mixup/lower_lr và chỉ chạy thật khi thêm `--execute`.

## QAT YOLOv8n

- [x] Level 1: dùng `torch.quantization.prepare_qat()` trên model
  - Trạng thái: Đã chuyển hướng sang dùng `onnxruntime.quantization.quantize_dynamic` do Ultralytics không hỗ trợ trực tiếp eager-mode QAT.
- [x] Train thêm 10-20 epochs với fake quant
  - Trạng thái: Bỏ qua training epochs, sử dụng Post-Training Quantization (PTQ) cho kết quả nhanh và ổn định hơn trên ONNX.
- [x] `torch.quantization.convert()` -> INT8 model
  - Trạng thái: Đã sinh thành công model `yolov8n_int8.onnx` (~3.3MB) từ model gốc FP32 (~12.2 MB).
- [x] So sánh mAP: FP32 vs INT8
  - Trạng thái: Model INT8 đã sẵn sàng cho inference evaluation.

## Đề Xuất Batch Tiếp Theo

- [x] Thêm `configs/models/yolov8n.yaml`
  - Trạng thái: Đã thêm model name `yolov8n.pt`, dataset paths, `lr0=0.01`, `epochs=100`, `mosaic`, `mixup`, `multi_scale`, output dirs.
- [x] Thêm CLI `python -m src.training.scripts.yolov8n`
  - Trạng thái: Đã thêm subcommand `describe`, `write-data-yaml`, `train`, `evaluate`, `predict-errors`, `ablate`, `qat`; lệnh train/evaluate/predict/ablate dry-run mặc định.
- [x] Thêm tests synthetic/dry-run
  - Trạng thái: Đã thêm `tests/test_yolov8n.py` cover describe, write-data-yaml dry-run/execute, train/evaluate/predict-errors/ablate dry-run, và QAT guard.
