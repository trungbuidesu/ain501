# Checklist YOLOv8n - Detection, Fine-Tune, QAT

File này theo dõi trạng thái YOLOv8n cho object detection. Quy ước:

- `[x]` đã có trong repo hoặc đã chạy pass ở mức ghi rõ trong trạng thái.
- `[ ]` chưa implement, chưa chạy thật, hoặc còn bị guard vì chưa chứng minh tương thích.
- `Trạng thái`: ghi rõ đang ở mức code-first, pilot, hay cần training/evaluation thật.

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
  - Trạng thái: CLI `describe` đã ghi confidence filtering + NMS; `predict-errors` đọc output prediction sau postprocess.

## Fine-Tune YOLOv8n

- [x] Dependency Ultralytics
  - Trạng thái: `pyproject.toml` đã khai báo `ultralytics>=8.0.0`; env `trungbd` hiện detect Ultralytics `8.4.37`.
- [x] Dataset class config
  - Trạng thái: `configs/datasets/object_detection_accessibility.yaml` đã có `classes.coco_subset`, `classes.custom_only`, `output.yolo_root`, `output.merged_root`, và `canonical_names_path`.
- [x] COCO/custom -> YOLO dataset helpers
  - Trạng thái: Đã có `convert`, `validate`, `merge-yolo`, `normalize-custom-yolo` trong `src.training.scripts.data_utils`; dùng để chuẩn bị layout `images/<split>` + `labels/<split>`.
- [x] Training config YAML cho YOLOv8n
  - Trạng thái: Đã có `configs/models/yolov8n.yaml` và CLI `write-data-yaml`; data YAML local/ignored tại `data/processed/object_detection_accessibility_merged/yolov8n_data.yaml` hiện trỏ đúng `train: images/train`, `val: images/val`, `nc: 16`.
- [x] Classes + augmentations + hyperparams trong training config
  - Trạng thái: Config full run khai báo `lr0=0.01`, `epochs=100`, `batch=16`, `imgsz=640`, `mosaic=1.0`, `mixup=0.1`, `multi_scale=true`; config pilot khai báo `epochs=5`, `device=cpu`, `fraction=0.005`, `multi_scale=false`.
- [x] Train trên COCO hoặc custom dataset
  - Trạng thái: Đã chạy pilot thật trên COCO YOLO train+val bằng CPU với `epochs=5`, `fraction=0.005`, `batch=16`, `mosaic=1.0`, `mixup=0.1`, `multi_scale=false`. Run artifact local/ignored nằm ở `D:\workspace\GitHub\ain501\runs\detect\runs\yolov8n\pilot_5e`. Đây chưa phải full training 50-100 epochs.
- [x] Base LR `0.01`, epochs `50-100`
  - Trạng thái: Đã cấu hình `lr0=0.01`, `epochs=100` trong `configs/models/yolov8n.yaml` và expose override qua CLI. Pilot dùng `epochs=5`; full production run 50-100 epochs vẫn chưa chạy.
- [x] Mosaic + mixup augmentation
  - Trạng thái: Đã cấu hình `mosaic=1.0`, `mixup=0.1`; pilot train thật đã chạy với hai augmentation này.
- [x] Multi-scale training
  - Trạng thái: Full config giữ `multi_scale=true` và CLI hỗ trợ `--multi-scale/--no-multi-scale`; pilot CPU tắt `multi_scale` vì lần thử đầu tạo lỗi output size không hợp lệ trên CPU/Ultralytics.

## Monitor / Evaluate / Error Analysis

- [x] Monitor loss curves
  - Trạng thái: Pilot train đã tạo `results.csv`, `args.yaml`, `weights/best.pt`, `weights/last.pt`. Train summary local/ignored nằm ở `D:\workspace\GitHub\ain501\reports\yolov8n\pilot_5e\train_summary.json`.
- [x] Monitor `mAP@0.5`
  - Trạng thái: Pilot train final `metrics/mAP50(B)=0.19236`; eval riêng trên val set `mAP50=0.19210`.
- [x] Monitor `mAP@0.5:0.95`
  - Trạng thái: Pilot train final `metrics/mAP50-95(B)=0.12639`; eval riêng trên val set `mAP50-95=0.12626`.
- [x] Evaluate trên val set -> report mAP
  - Trạng thái: Đã chạy `evaluate --execute` với `weights/best.pt` của pilot. Eval summary local/ignored nằm ở `D:\workspace\GitHub\ain501\reports\yolov8n\pilot_5e\eval_summary.json`.
- [x] Error analysis: xem predictions sai, tìm pattern
  - Trạng thái: Đã chạy `predict-errors --execute --max-samples 25` trên val images và xuất CSV local/ignored tại `D:\workspace\GitHub\ain501\reports\yolov8n\pilot_5e\errors.csv`. Đây là sample nhỏ để kiểm tra pipeline, chưa phải phân tích lỗi đầy đủ.
- [x] Optional ablation study: thử bỏ/thêm augmentation, thay LR
  - Trạng thái: Đã có CLI `ablate` tạo matrix baseline/no_mosaic/no_mixup/lower_lr và chỉ chạy thật khi thêm `--execute`; chưa chạy ablation thật.

## QAT YOLOv8n

- [x] QAT compatibility guard
  - Trạng thái: Đã có CLI `qat` report `supported=false` để tránh force QAT lên Ultralytics YOLOv8n khi compatibility chưa được chứng minh.
- [ ] Level 1: dùng `torch.quantization.prepare_qat()` trên model
  - Trạng thái: Chưa chạy actual `prepare_qat()` trên Ultralytics YOLOv8n; giữ mở/guarded để tránh tạo INT8 artifact hoặc metric không đáng tin cậy.
- [ ] Train thêm 10-20 epochs với fake quant
  - Trạng thái: Chưa có QAT training loop hoặc config epochs riêng.
- [ ] `torch.quantization.convert()` -> INT8 model
  - Trạng thái: Chưa có export/convert artifact INT8.
- [ ] So sánh mAP: FP32 vs INT8
  - Trạng thái: Chưa có report so sánh metric giữa baseline FP32 và INT8.

## Batch Hiện Tại

- [x] Thêm pilot defaults vào `configs/models/yolov8n.yaml`
  - Trạng thái: Đã thêm preset `pilot` với `epochs=5`, `name=pilot_5e`, `device=cpu`, `fraction=0.005`, `workers=0`, `patience=2`, `exist_ok=true`, và summary/report paths.
- [x] Mở rộng CLI `python -m src.training.scripts.yolov8n`
  - Trạng thái: `train`, `evaluate`, và `predict-errors` hỗ trợ `--preset`; `train/evaluate` hỗ trợ summary JSON; `train` hỗ trợ `--fraction`, `--workers`, `--patience`, `--exist-ok`.
- [x] Thêm tests synthetic/dry-run
  - Trạng thái: `tests/test_yolov8n.py` cover describe, write-data-yaml dry-run/execute, train/evaluate/predict-errors/ablate dry-run, QAT guard, pilot preset, và summary writer.
