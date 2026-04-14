# Object Detection Dataset (Bộ dữ liệu phát hiện đối tượng)

## Mục Tiêu

Object detection là tác vụ core của Phase 0 để hỗ trợ nhận biết vật cản và vùng an toàn.

## Cấu Hình, Nguồn Và Class

- Config: `configs/datasets/object_detection_accessibility.yaml`
- Nguồn bootstrap: COCO 2017
- Mở rộng tùy chọn: custom capture cho các class accessibility còn thiếu
- BBox source format: `coco_xywh_pixels`, output format: YOLO (`images/<split>`, `labels/<split>`)

## Nguồn Tải

- Trang chủ: [https://cocodataset.org/](https://cocodataset.org/)
- URL tải trực tiếp theo config:
  - `http://images.cocodataset.org/zips/train2017.zip`
  - `http://images.cocodataset.org/zips/val2017.zip`
  - `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

## Thông Số Quan Trọng

| Nhóm | Thông số | Giá trị |
| --- | --- | --- |
| Metadata | `task` | `object_detection` |
| Metadata | `phase` | `0.3` |
| Format | `bbox_format` | `coco_xywh_pixels` |
| Output | `yolo_root` | `data/processed/object_detection_accessibility` |
| Output | `merged_root` | `data/processed/object_detection_accessibility_merged` |
| Class | `coco_subset` | 16 class |
| Class | `custom_only` | 6 class (`door`, `stairs`, `crosswalk_signal`, `elevator`, `curb`, `sidewalk_obstacle`) |
| Custom workflow | `target_images` | `200-500` |
| Custom workflow | `export_format` | `ultralytics_yolo` |

### Class Strategy

- `coco_subset`: class có sẵn trong COCO để bootstrap nhanh.
- `custom_only`: class cần bổ sung bằng dữ liệu thực tế (ví dụ `elevator`, `crosswalk_signal`, `sidewalk_obstacle`).
- `canonical_names_path`: `data/processed/object_detection_accessibility/classes.txt` là nguồn chuẩn cho remap class id.

## Cấu Trúc Input/Output

- Input roots:
  - `data/external/coco2017`
  - `data/custom/annotation_exports`
- Output roots:
  - `data/processed/object_detection_accessibility`
  - `data/processed/object_detection_accessibility_merged`
- Artifact chính:
  - `classes.txt`
  - ảnh + nhãn YOLO đã normalize
  - kết quả validate dataset

## Workflow Chi Tiết

### Bước 1: Chuẩn hóa class chuẩn

```bash
python -m src.training.scripts.data_utils write-classes --dry-run
python -m src.training.scripts.data_utils write-classes --execute
```

### Bước 2: Convert COCO sang YOLO

```bash
python -m src.training.scripts.data_utils convert --coco-json data/external/coco2017/annotations/instances_train2017.json --image-root data/external/coco2017/train2017 --output-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt --split train --dry-run
python -m src.training.scripts.data_utils convert --coco-json data/external/coco2017/annotations/instances_val2017.json --image-root data/external/coco2017/val2017 --output-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt --split val --execute
```

### Bước 3: Validate YOLO layout

```bash
python -m src.training.scripts.data_utils validate --dataset-root data/processed/object_detection_accessibility --classes data/processed/object_detection_accessibility/classes.txt
```

### Bước 4: Ingest dữ liệu custom (nếu có)

```bash
python -m src.training.scripts.data_utils normalize-custom-yolo --export-root data/custom/annotation_exports/export_001 --output-root data/custom/annotation_exports/export_001_normalized --classes data/processed/object_detection_accessibility/classes.txt --source-format ultralytics_yolo --dry-run
python -m src.training.scripts.data_utils merge-yolo --sources data/custom/annotation_exports/export_001_normalized data/processed/object_detection_accessibility --output-root data/processed/object_detection_accessibility_merged --classes data/processed/object_detection_accessibility/classes.txt --dry-run
```

Chỉ thêm `--execute` sau khi đã xác nhận path, dung lượng lưu trữ và yêu cầu riêng tư.

## Tiêu Chí Hoàn Thành

- Có `classes.txt` canonical và mapping class id thống nhất.
- Dataset root có đủ cặp `images/<split>` và `labels/<split>`.
- Lệnh `validate` pass, không lỗi bbox/class id/missing pair.
- Nếu dùng custom data: đã qua review riêng tư trước khi merge.

## Lỗi Thường Gặp

- Sai class order giữa `classes.txt` và export custom -> remap sai label.
- Export custom thiếu split rõ ràng -> cần normalize trước khi merge.
- BBox ra ngoài `[0, 1]` sau convert -> kiểm tra input annotation nguồn.
- Chạy `--execute` quá sớm -> ghi đè output không mong muốn.

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
