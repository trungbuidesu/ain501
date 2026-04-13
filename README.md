# ain501

AIN501 là dự án hỗ trợ phát triển ứng dụng AI và agentic theo hướng sạch,
module hóa, dễ kiểm thử và dễ mở rộng.

## Cấu Trúc Dự Án

- `src/`: Mã nguồn chính.
- `models/`: Định nghĩa model, kiến trúc, trọng số và checkpoint.
- `configs/`: File cấu hình YAML/JSON.
- `data/`: Dữ liệu local, không commit vào git.
- `tests/`: Unit test và integration test.
- `docs/`: Tài liệu dự án.
- `agents/`: Tài liệu và logic dành cho agent.

## Bắt Đầu

### 1. Yêu Cầu

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) hoặc [Anaconda](https://www.anaconda.com/).
- Python 3.10+.

### 2. Tạo Môi Trường

Tạo và kích hoạt conda environment:

```bash
conda create --name ain501 python=3.10
conda activate ain501
```

### 3. Cài Đặt

Cài project ở chế độ editable kèm dependency cho development:

```bash
pip install -e ".[dev]"
```

Nếu PowerShell không nhận `python` hoặc `conda` qua PATH, dùng trực tiếp
interpreter của conda environment trong project:

```powershell
$AIN501_PY="C:\Users\bdtrung29.1\miniconda3\envs\ain501\python.exe"
& $AIN501_PY -m pip install -e ".[dev]"
```

## Quy Ước Phát Triển

Quy tắc commit và kiểm tra code chi tiết nằm trong [agents/rules.md](agents/rules.md).

### Kiểm Tra Nhanh

- **Format**: `black .`
- **Lint**: `ruff check .`
- **Type check**: `mypy .`
- **Test**: `pytest`

Fallback PowerShell khi `python` không có trong PATH:

```powershell
$AIN501_PY="C:\Users\bdtrung29.1\miniconda3\envs\ain501\python.exe"
& $AIN501_PY -m black --check .
& $AIN501_PY -m ruff check .
& $AIN501_PY -m mypy .
& $AIN501_PY -m pytest -p no:cacheprovider
```
