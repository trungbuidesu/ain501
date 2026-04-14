# Action Recognition Dataset (Bộ dữ liệu nhận diện hành động)

## Mục Tiêu

Action recognition là tác vụ video core để nhận diện mẫu hành vi từ chuỗi frame.

## Cấu Hình Và Nguồn

- Config: `configs/datasets/action_accessibility.yaml`
- Nguồn bootstrap: UCF-101 với bộ split chính thức
- Tham số chuỗi frame được điều khiển từ config

## Workflow Cơ Bản

```bash
python -m src.training.scripts.data_utils build-action-manifest
python -m src.training.scripts.data_utils validate-video-manifest --manifest-csv data/processed/action_accessibility/manifest.csv
```

Chỉ dùng `--execute` khi đã xác nhận đầy đủ path dataset và split files local.

## Input Và Output

- Input root: `data/external/ucf101`
- Output root: `data/processed/action_accessibility`
- Artifact chính: `manifest.csv`

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
