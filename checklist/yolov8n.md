# Checklist YOLOv8n - Detection, Fine-Tune, QAT

File này theo dõi trạng thái YOLOv8n cho object detection. Quy ước:

- `[x]` đã có trong repo hoặc đã được chuẩn bị ở mức dependency/config/helper.
- `[ ]` chưa implement hoặc chưa chạy thật.
- `Trạng thái`: ghi rõ đang ở mức code-first, dataset helper, hay cần training/evaluation thật.

## YOLOv8n Architecture Understanding

- [ ] Backbone: CSPDarknet / C2f-style feature extractor
  - Trạng thái: Chưa có tài liệu hoặc module inspect YOLOv8n trong repo. Hiện mới có dependency `ultralytics>=8.0.0`.
- [ ] Neck: PANet/FPN-style feature fusion
  - Trạng thái: Chưa có ghi chú kiến trúc hoặc CLI `describe` cho YOLOv8n.
- [ ] Head: decoupled detection head
  - Trạng thái: Chưa có phần đọc/ghi chú head YOLOv8n trong repo.
- [ ] Anchor-free detection
  - Trạng thái: Chưa được document trong checklist/docs dự án.
- [ ] Loss composition: box + cls + dfl
  - Trạng thái: Chưa có training wrapper hoặc note loss composition cho YOLOv8n.
- [ ] NMS postprocessing
  - Trạng thái: Chưa có note hoặc error-analysis tool đọc prediction/NMS output.

## Fine-Tune YOLOv8n

- [x] Dependency Ultralytics
  - Trạng thái: `pyproject.toml` đã khai báo `ultralytics>=8.0.0`; env `trungbd` hiện detect được Ultralytics.
- [x] Dataset class config
  - Trạng thái: `configs/datasets/object_detection_accessibility.yaml` đã có `classes.coco_subset`, `classes.custom_only`, `output.yolo_root`, `output.merged_root`, và `canonical_names_path`.
- [x] COCO/custom -> YOLO dataset helpers
  - Trạng thái: Đã có `convert`, `validate`, `merge-yolo`, `normalize-custom-yolo` trong `src.training.scripts.data_utils`; dùng để chuẩn bị layout `images/<split>` + `labels/<split>`.
- [ ] Training config YAML cho YOLOv8n
  - Trạng thái: Chưa có `configs/models/yolov8n.yaml` hoặc Ultralytics `data.yaml`/train config sinh tự động từ object detection config.
- [ ] Classes + augmentations + hyperparams trong training config
  - Trạng thái: Chưa khai báo YOLOv8n defaults: `lr0=0.01`, `epochs=50-100`, mosaic, mixup, multi-scale.
- [ ] Train trên COCO hoặc custom dataset
  - Trạng thái: Chưa có CLI/wrapper training YOLOv8n. Local hiện chỉ chắc chắn có COCO val processed; COCO train/custom data có thể chưa sẵn sàng.
- [ ] Base LR `0.01`, epochs `50-100`
  - Trạng thái: Chưa được cấu hình trong model config.
- [ ] Mosaic + mixup augmentation
  - Trạng thái: Chưa được cấu hình cho YOLOv8n training.
- [ ] Multi-scale training
  - Trạng thái: Chưa được cấu hình cho YOLOv8n training.

## Monitor / Evaluate / Error Analysis

- [ ] Monitor loss curves
  - Trạng thái: Chưa có training run YOLOv8n hoặc TensorBoard/Ultralytics run artifact được tạo.
- [ ] Monitor `mAP@0.5`
  - Trạng thái: Chưa có evaluation wrapper/report.
- [ ] Monitor `mAP@0.5:0.95`
  - Trạng thái: Chưa có evaluation wrapper/report.
- [ ] Evaluate trên val set -> report mAP
  - Trạng thái: Chưa có CLI `eval` hoặc report artifact cho YOLOv8n.
- [ ] Error analysis: xem predictions sai, tìm pattern
  - Trạng thái: Chưa có command xuất false positives / false negatives / low-confidence sample gallery.
- [ ] Optional ablation study: thử bỏ/thêm augmentation, thay LR
  - Trạng thái: Chưa có ablation config matrix hoặc script runner.

## QAT YOLOv8n

- [ ] Level 1: dùng `torch.quantization.prepare_qat()` trên model
  - Trạng thái: Chưa implement. Cần kiểm tra khả năng tương thích trực tiếp với Ultralytics YOLOv8n trước khi chạy thật.
- [ ] Train thêm 10-20 epochs với fake quant
  - Trạng thái: Chưa có QAT training loop hoặc config epochs riêng.
- [ ] `torch.quantization.convert()` -> INT8 model
  - Trạng thái: Chưa có export/convert artifact INT8.
- [ ] So sánh mAP: FP32 vs INT8
  - Trạng thái: Chưa có report so sánh metric giữa baseline FP32 và INT8.

## Đề Xuất Batch Tiếp Theo

- [ ] Thêm `configs/models/yolov8n.yaml`
  - Trạng thái đề xuất: Chứa model name `yolov8n.pt`, dataset paths, `lr0=0.01`, `epochs=100` mặc định, `mosaic`, `mixup`, `multi_scale`, output dirs.
- [ ] Thêm CLI `python -m src.training.scripts.yolov8n`
  - Trạng thái đề xuất: Subcommand `describe`, `write-data-yaml`, `train`, `evaluate`, `predict-errors`, `ablate`, `qat`.
- [ ] Thêm tests synthetic/dry-run
  - Trạng thái đề xuất: Không cần dataset thật; verify config generation, CLI parse, architecture summary, và QAT guard nếu API không hỗ trợ.
