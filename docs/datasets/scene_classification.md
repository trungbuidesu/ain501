# Scene Classification Dataset (Bộ dữ liệu phân loại bối cảnh)

## Mục Tiêu

Scene classification là tác vụ core để suy luận bối cảnh môi trường xung quanh.

## Cấu Hình, Nguồn Và Class Subset

- Config: `configs/datasets/scene_accessibility.yaml`
- Nguồn bootstrap: Places365
- Cơ chế lọc class và cân bằng dữ liệu được điều khiển từ config
- `max_images_per_class` dùng để tránh lệch phân phối và tránh bùng nổ dung lượng.

## Nguồn Tải

- Trang dự án: [https://github.com/CSAILVision/places365](https://github.com/CSAILVision/places365)
- Variant theo config: `places365_standard_256`
- Metadata bắt buộc:
  - `data/external/places365/categories_places365.txt`
  - file list như `places365_val.txt`

## Thông Số Quan Trọng

| Nhóm | Thông số | Giá trị |
| --- | --- | --- |
| Metadata | `task` | `scene_classification` |
| Metadata | `phase` | `0.5` |
| Balancing | `max_images_per_class` | `1000` |
| Balancing | `split_seed` | `501` |
| Class | `places_subset` | 16 class |
| Class | `custom_or_proxy` | `sidewalk`, `bus_stop` |
| Output | `subset_root` | `data/processed/scene_accessibility` |
| Output | `manifest` | `data/processed/scene_accessibility/manifest.csv` |

## Cấu Trúc Input/Output

- Input roots:
  - `data/external/places365`
  - `data/external/places365/categories_places365.txt`
  - file list như `places365_val.txt` (khi tạo subset từ file list)
- Output root:
  - `data/processed/scene_accessibility`
- Artifact chính:
  - `manifest.csv` cho subset đã lọc và cân bằng

## Workflow Chi Tiết

### Bước 1: Kiểm tra path và metadata nguồn

```bash
python -m src.training.scripts.data_utils verify-dataset-paths --tasks scene_classification --skip-downloads
```

### Bước 2: Tạo subset cân bằng

```bash
python -m src.training.scripts.data_utils build-scene-subset --file-list data/external/places365/places365_val.txt --dry-run
python -m src.training.scripts.data_utils build-scene-subset --file-list data/external/places365/places365_val.txt --execute
```

Chỉ thêm `--execute` sau khi xác nhận file dataset đầu vào và output location.

## Lưu Ý Class Và Mapping

- `places_subset` là danh sách class dùng trực tiếp từ Places365.
- `custom_or_proxy` ghi rõ class chưa có trực tiếp (ví dụ `sidewalk`, `bus_stop`) cần custom capture hoặc proxy mapping có kiểm soát.

## Tiêu Chí Hoàn Thành

- Manifest sinh ra với phân phối class cân bằng theo config.
- Không có class ngoài danh sách cấu hình.
- Output có thể dùng ngay cho pipeline train/eval scene model.

## Lỗi Thường Gặp

- Thiếu `categories_places365.txt` -> không resolve được class id -> name.
- File list không khớp root ảnh -> manifest rỗng hoặc thiếu nặng.
- Cấu hình `max_images_per_class` quá cao -> tăng chi phí IO không cần thiết.

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
