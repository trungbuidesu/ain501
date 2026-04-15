# Checklist all-MiniLM-L6-v2 - Text Embedding & Retrieval

File này theo dõi trạng thái triển khai all-MiniLM-L6-v2. Quy ước:

- `[x]` đã có code hoặc đã verify ở mức smoke test.
- `[/]` đang thực hiện.
- `[ ]` chưa hoàn thành trên dữ liệu domain thật.
- `Trạng thái`: nêu rõ mức smoke/prod và artifact liên quan.

## 1. Kiến trúc và baseline

- [x] Load pretrained model từ Sentence-Transformers
  - Trạng thái: CLI `describe`/`evaluate-*` dùng `sentence-transformers/all-MiniLM-L6-v2`.
- [x] Mô tả kiến trúc distilled BERT + pooling strategy
  - Trạng thái: Cập nhật tài liệu tại `docs/models/all_minilm.md`.
- [x] Baseline cosine similarity test
  - Trạng thái: Có subcommand `evaluate-similarity`, so sánh cặp tương đồng/không tương đồng.

## 2. Dữ liệu contrastive pairs

- [x] Tạo positive/negative pairs từ caption templates
  - Trạng thái: Có subcommand `build-pairs`, sinh CSV `anchor,positive,negative,source,split`.
- [x] Split train/val/test có kiểm soát seed
  - Trạng thái: `assign_splits` theo `train_ratio/val_ratio`, deterministic theo `seed`.
- [ ] Mở rộng template/pairs bằng dữ liệu domain thật
  - Trạng thái: Chưa chạy full domain dataset; hiện dùng template bootstrap.

## 3. Fine-tune contrastive

- [x] InfoNCE training flow
  - Trạng thái: `fine-tune --objective infonce` dùng `MultipleNegativesRankingLoss`.
- [x] Optional triplet training flow
  - Trạng thái: `fine-tune --objective triplet` dùng `TripletLoss`.
- [x] Train 5-10 epochs + checkpoint
  - Trạng thái: Config mặc định 8 epochs, lưu model vào `models/text/minilm_l6/<run_name>/`.
- [x] Logging train summary
  - Trạng thái: Ghi `summary.json` trong `reports/minilm_l6/` + TensorBoard scalar trước/sau.

## 4. Retrieval quality

- [x] Đo Recall@k, MRR trước/sau fine-tune
  - Trạng thái: Có `evaluate-retrieval` và báo cáo baseline/final trong `fine-tune`.
- [ ] Đạt cải thiện ổn định trên dữ liệu thật
  - Trạng thái: Chưa benchmark production corpus; mới có pipeline và smoke eval.

## 5. Export ONNX và Benchmark CPU

- [x] Export ONNX embedding model
  - Trạng thái: `export-onnx` xuất `models/minilm_l6.onnx` (encoder + mean pooling + normalize).
- [x] Benchmark CPU per-embedding latency
  - Trạng thái: `benchmark` trả `latency_ms_p50/p95` và cờ `target_under_5ms`.
- [ ] Xác nhận `< 5ms` trên máy target production
  - Trạng thái: Chưa có số liệu chính thức trên target CPU deployment.

## 6. Test / docs / maintenance

- [x] Test cho pair builder và retrieval metric
  - Trạng thái: `tests/test_minilm_l6.py` cover generation/split/csv/retrieval smoke.
- [x] Tài liệu model được cập nhật
  - Trạng thái: `docs/models/all_minilm.md` đã có quy trình và lệnh CLI.
- [/] Theo dõi tiến độ thực nghiệm domain
  - Trạng thái: Checklist này sẽ tiếp tục update khi có run thật.
