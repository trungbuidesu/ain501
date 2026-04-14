# Action Recognition Dataset (Bộ dữ liệu nhận diện hành động)

## Mục Tiêu

Action recognition là tác vụ video core để nhận diện mẫu hành vi từ chuỗi frame.

## Cấu Hình, Nguồn Và Sampling

- Config: `configs/datasets/action_accessibility.yaml`
- Nguồn bootstrap: UCF-101 với bộ split chính thức
- Tham số chuỗi frame được điều khiển từ config
- Mặc định hiện tại:
  - `frame_sequence_length: 16`
  - `frame_stride: 2`
  - `split_seed: 501`

## Nguồn Tải

- Trang chính thức: [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)
- Theo config, dữ liệu đặt local:
  - video dưới `data/external/ucf101/UCF-101`
  - split files dưới `data/external/ucf101/ucfTrainTestlist`

## Thông Số Quan Trọng

| Nhóm | Thông số | Giá trị |
| --- | --- | --- |
| Metadata | `task` | `action_recognition` |
| Metadata | `phase` | `0.4` |
| Sampling | `frame_sequence_length` | `16` |
| Sampling | `frame_stride` | `2` |
| Sampling | `split_seed` | `501` |
| Class | `public_bootstrap` | 20 class |
| Class | `custom_required` | 8 class |
| Output | `manifest` | `data/processed/action_accessibility/manifest.csv` |
| Output | `split_manifest` | `data/processed/action_accessibility/splits.csv` |

## Cấu Trúc Input/Output

- Input root:
  - `data/external/ucf101/UCF-101` (videos)
  - `data/external/ucf101/ucfTrainTestlist` (official split lists)
- Output roots:
  - `data/processed/action_accessibility/manifest.csv`
  - `data/processed/action_accessibility/splits.csv` (khi tạo split)
- Artifact chính:
  - manifest class/video/split dùng cho pipeline train và validation

## Workflow Chi Tiết

### Bước 1: Kiểm tra cấu trúc dữ liệu video

```bash
python -m src.training.scripts.data_utils verify-dataset-paths --tasks action_recognition --skip-downloads
```

### Bước 2: Tạo manifest từ class-folder videos

```bash
python -m src.training.scripts.data_utils build-action-manifest --dry-run
python -m src.training.scripts.data_utils build-action-manifest --execute
```

### Bước 3: Validate manifest

```bash
python -m src.training.scripts.data_utils validate-video-manifest --manifest-csv data/processed/action_accessibility/manifest.csv
```

Chỉ dùng `--execute` khi đã xác nhận đầy đủ path dataset và split files local.

## Lưu Ý Theo Use Case

- Nhóm class `public_bootstrap` dùng để có pipeline hoạt động ngay.
- Nhóm `custom_required` là mục tiêu extension khi có dữ liệu thực tế.
- `approaching` được ghi nhận là derived task ở phase sau, không block Phase 0.

## Tiêu Chí Hoàn Thành

- Manifest được tạo thành công với số dòng hợp lệ.
- `validate-video-manifest` pass:
  - file video tồn tại,
  - label thuộc tập cho phép,
  - không có split/group leakage (nếu bật check liên quan).
- Sampling settings được giữ ổn định giữa các run để so sánh công bằng.

## Lỗi Thường Gặp

- Sai root `UCF-101` hoặc thiếu `ucfTrainTestlist` -> không map được split.
- Video corrupt/không mở được -> cần bật check mở video khi cần điều tra.
- Class folder lệch tên chuẩn -> label sai giữa manifest và config.

## Tham Chiếu

- `docs/data_pipeline.md`
- `configs/datasets/README.md`
