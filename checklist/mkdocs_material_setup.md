# Checklist MkDocs Material Setup

File này dùng để theo dõi tiến độ dựng docs tạm với MkDocs Material. Quy ước:

- `[x]` đã implement trong repo hoặc đã hoàn thành thực tế.
- `[ ]` chưa hoàn thành.
- `Trạng thái`: ghi chú ngắn để biết đang ở mức scaffold, local verify, hay production-ready.

## MkDocs Configuration

- [x] Tạo `mkdocs.yml` tại root với `theme: material`
  - Trạng thái: Đã tạo nav tạm gồm Home, nhóm Docs, và Existing -> Data Pipeline.
- [x] Đưa `docs/data_pipeline.md` vào navigation
  - Trạng thái: Đã map trong nav và giữ nguyên nội dung file hiện có.

## Dummy Pages

- [x] Tạo landing page `docs/index.md`
  - Trạng thái: Đã có nội dung placeholder cho scaffold docs.
- [x] Tạo các dummy pages (`docs/general/overview.md`, `docs/general/experiments.md`, `docs/general/models.md`)
  - Trạng thái: Đã tạo nội dung placeholder để test điều hướng và đã map trong `mkdocs.yml`.

## Local Validation

- [x] Chạy `mkdocs serve` để kiểm tra render local
  - Trạng thái: Đã cài `mkdocs-material` qua dev extra; lần kiểm tra gần nhất dùng env `trungbd`. Trước đó đã chạy `python -m mkdocs serve --dev-addr 127.0.0.1:8001 --no-livereload`, request local trả `status=200`, sau đó đã stop server.
- [x] Chạy `mkdocs build` để kiểm tra build output
  - Trạng thái: Đã chạy `python -m mkdocs build --strict --site-dir %TEMP%/ain501_mkdocs_check`, build pass. `site/` được git-ignore cho build mặc định.
