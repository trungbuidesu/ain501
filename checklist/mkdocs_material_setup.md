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
- [x] Tạo các dummy pages (`docs/overview.md`, `docs/experiments.md`, `docs/models.md`)
  - Trạng thái: Đã tạo nội dung placeholder để test điều hướng.

## Local Validation

- [ ] Chạy `mkdocs serve` để kiểm tra render local
  - Trạng thái: Chờ verify trong môi trường hiện tại.
- [ ] Chạy `mkdocs build` để kiểm tra build output
  - Trạng thái: Chờ verify trong môi trường hiện tại.
