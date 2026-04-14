# Object Detection Dataset (Bộ dữ liệu phát hiện đối tượng)

## Mục Tiêu

Object detection là tác vụ core của Phase 0 để hỗ trợ nhận biết vật cản và vùng an toàn.

## Cấu Hình Và Nguồn

- Config: `configs/datasets/object_detection_accessibility.yaml`
- Nguồn bootstrap: COCO 2017
- Mở rộng tùy chọn: custom capture cho các class accessibility còn thiếu

## Workflow Cơ Bản

```bash
python -m src.training.scripts.data_utils write-classes
python -m src.training.scripts.data_utils convert --coco-json data/external/coco2017/annotations/instances_val2017.json --image-root data/external/coco2017/val2017 --output-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt --split val
python -m src.training.scripts.data_utils validate --dataset-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt
python -m src.training.scripts.data_utils normalize-custom-yolo --export-root data/custom/annotation_exports/export_001 --output-root data/custom/annotation_exports/export_001_normalized --classes data/processed/object_detection_accessibility/classes.txt --source-format ultralytics_yolo --dry-run
```

Chỉ thêm `--execute` sau khi đã xác nhận path, dung lượng lưu trữ và yêu cầu riêng tư.

## Input Và Output

- Input roots: `data/external/coco2017`, `data/custom/annotation_exports`
- Output root: `data/processed/object_detection_accessibility`
- Artifact chính: `classes.txt`, labels/images chuẩn YOLO, báo cáo validate

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
