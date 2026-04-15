# Hệ Thống Nhúng Từ Vựng Ngữ Nghĩa: all-MiniLM-L6-v2

`all-MiniLM-L6-v2` là mô hình embedding văn bản nhẹ (~22M params) dùng cho truy xuất
ngữ nghĩa trong pipeline RAG của AIN501.

## Kiến trúc

- Backbone: MiniLM thuộc họ distilled BERT encoder (`6` Transformer layers).
- Hidden size: `384`.
- Pooling mặc định: **mean pooling** trên token embeddings + attention mask.
- Chuẩn hóa vector: L2 normalization trước khi tính cosine similarity.

Kiến trúc này cân bằng giữa chất lượng semantic retrieval và độ trễ CPU thấp.

## Cấu hình mặc định

File cấu hình: `configs/models/minilm_l6.yaml`

- Model ID: `sentence-transformers/all-MiniLM-L6-v2`
- Max sequence length: `128`
- Objective mặc định khi fine-tune: `InfoNCE` (`MultipleNegativesRankingLoss`)
- Fine-tune mặc định: `8` epochs, batch size `32`, lr `2e-5`
- ONNX output: `models/minilm_l6.onnx`

## CLI

Module chính: `python -m src.training.scripts.minilm_l6`

- `describe`: in thông tin distilled BERT + pooling strategy.
- `build-pairs`: tạo positive/negative pairs từ caption templates.
- `evaluate-similarity`: đo cosine trên các câu tương đồng/không tương đồng.
- `fine-tune`: train contrastive (`infonce` hoặc `triplet`) 5-10 epochs.
- `evaluate-retrieval`: đo Recall@k, MRR trước/sau fine-tune.
- `export-onnx`: xuất encoder + pooling sang ONNX.
- `benchmark`: đo latency CPU (`p50/p95`) cho embedding inference.

## Quy trình đề xuất

1. Tạo pairs template:
   `python -m src.training.scripts.minilm_l6 build-pairs`
2. Baseline quality:
   `python -m src.training.scripts.minilm_l6 evaluate-similarity`
3. Fine-tune:
   `python -m src.training.scripts.minilm_l6 fine-tune --pairs-csv data/processed/text_embedding/minilm_pairs.csv`
4. So sánh retrieval:
   `python -m src.training.scripts.minilm_l6 evaluate-retrieval --pairs-csv data/processed/text_embedding/minilm_pairs.csv --split val --k 5`
5. Export ONNX:
   `python -m src.training.scripts.minilm_l6 export-onnx --output models/minilm_l6.onnx`
6. Benchmark CPU:
   `python -m src.training.scripts.minilm_l6 benchmark --runtime onnxruntime --onnx-path models/minilm_l6.onnx`

## Tiêu chí chất lượng

- Cosine similarity trung bình của cặp tương đồng phải cao hơn cặp không tương đồng.
- Retrieval sau fine-tune cần so sánh với baseline theo `Recall@k` và `MRR`.
- Mục tiêu deploy CPU: `< 5ms` / embedding (p50).
