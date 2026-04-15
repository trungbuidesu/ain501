# Phase 3 — RAG Lite + Orchestrator

## Tổng quan

- **Knowledge base:** MiniLM-L6 ONNX (xem [all-MiniLM-L6](../models/all_minilm.md), `models/minilm_l6.onnx`) + FAISS `IndexFlatIP` trên vector đã chuẩn hóa L2 (tương đương cosine).
- **Dữ liệu:** JSON UTF-8 trong `data/knowledge/` (`hazard_patterns.json`, `spatial_templates.json`, `relationship_patterns.json`), ~80 mục.
- **Orchestrator:** `src/caption/orchestrator.py` tạo chuỗi truy vấn tiếng Anh từ detection + Tier 1 agents + `scene_type`, gọi RAG; nếu độ tương đồng tốt nhất < `min_similarity` hoặc RAG tắt → fallback `SpatialTemplateEngine` (không duplicate logic mô tả bbox).
- **Cấu hình:** `configs/app.yaml` (khối `rag:` cùng file với capture/models/tier1).

## Bật / tắt RAG

Trong `configs/app.yaml`:

```yaml
rag:
  enabled: true   # false: chỉ template + Tier 1 merge như trước
  model_path: models/minilm_l6.onnx
  knowledge_dir: data/knowledge
  top_k: 3
  min_similarity: 0.3
  dedup_similarity: 0.8
```

Nếu khởi tạo RAG lỗi (thiếu ONNX, v.v.), pipeline in cảnh báo và chạy **template only**.

## Chạy

```bash
python -m src.app
```

## Checkpoint ví dụ (mục tiêu UX)

| Nguồn | Ví dụ |
|--------|--------|
| Template | Phía trước gần có person; phía trước gần có dog (hai câu tách) |
| RAG | Một người đang dắt chó (một câu tự nhiên hơn khi retrieval khớp `person dog leash`) |

Có thể tinh chỉnh thêm entry trong JSON nếu retrieval chưa khớp tình huống thực tế.

## Kiểm thử

```bash
pytest tests/test_rag.py -q
```

Smoke ONNX (khi có file model):

```bash
pytest tests/test_rag.py::test_minilm_embedder_smoke -q
```
