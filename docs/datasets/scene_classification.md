# Scene Classification Dataset (Bộ dữ liệu phân loại bối cảnh)

## Mục Tiêu

Scene classification là tác vụ core để suy luận bối cảnh môi trường xung quanh.

## Cấu Hình Và Nguồn

- Config: `configs/datasets/scene_accessibility.yaml`
- Nguồn bootstrap: Places365
- Cơ chế lọc class và cân bằng dữ liệu được điều khiển từ config

## Workflow Cơ Bản

```bash
python -m src.training.scripts.data_utils build-scene-subset --file-list data/external/places365/places365_val.txt
```

Chỉ thêm `--execute` sau khi xác nhận file dataset đầu vào và output location.

## Input Và Output

- Input root: `data/external/places365`
- Output root: `data/processed/scene_accessibility`
- Artifact chính: `manifest.csv` cho subset đã cân bằng

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
