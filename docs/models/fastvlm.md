# Kiến Trúc Ngôn Ngữ Tầm Nhìn: FastVLM-0.5B

> [!NOTE] Bản Nháp Tài Liệu (Draft Note)
> Trang tài liệu này đang trong tiến trình soạn thảo (Work in Progress).

Đặc tả dự kiến cho mô hình nhúng FastVLM-0.5B (khoảng 500M tham số, dung lượng ~300MB) phục vụ tác vụ sinh mô tả hình ảnh/video cấp 2 (Tier 2 VLM captioning). Trang sẽ sớm được bổ sung lưu đồ thiết kế và báo cáo thông số huấn luyện chi tiết.

## Week 5 verification (inference only)

**Artifact:** [apple/FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B) (BF16 weights, custom `trust_remote_code`). Download with `huggingface-cli download apple/FastVLM-0.5B`. Optional env: `FASTVLM_MODEL_ID` to override the default repo id.

**CLI:**

```text
python -m src.training.scripts.verify_prebuilt_week5 fastvlm --synthetic --limit 20 --cpu
python -m src.training.scripts.verify_prebuilt_week5 fastvlm --skip-model
```

`--skip-model` writes `reports/prebuilt_week5/fastvlm_benchmark.json` with status `skipped` when weights are not yet available (CI-friendly).

**Device:** `--device auto` (default) uses `src.utils.device.resolve_device`: on a machine with **Intel Arc** (e.g. A770) and a PyTorch build with `torch.xpu`, inference runs on **XPU** with float32; otherwise CUDA (float16) or CPU. Use `--cpu` to force CPU regardless of `--device`.

**Outputs:**

- `reports/prebuilt_week5/fastvlm_captions.jsonl` — per-image caption, prompt, TTFT, end-to-end latency, `quality_rubric` placeholder for human review.
- `reports/prebuilt_week5/fastvlm_benchmark.json` — device, dtype, p50/p95 **TTFT** (streaming first token) and **e2e** latency, optional RAM note.

**Implementation note:** Inference follows the Hugging Face model card (chat template with `<image>` placeholder, image token id `-200`, `model.get_vision_tower().image_processor`, `trust_remote_code=True`).

### Prompt templates (accessibility-oriented)

Use short, safety-biased prompts; log the exact string in reports.

1. **General scene (default in script):** `Describe the image briefly for a blind traveler.`
2. **Hazards first:** `List any hazards or obstacles visible, then give a one-sentence scene summary.`
3. **Text / signs:** `Describe the scene, then mention any readable text or signs you see.`
4. **Indoor navigation:** `Describe indoor layout cues (doors, stairs, furniture) useful for orientation.`

### Optional quantization (pilot)

If no suitable INT8/INT4 release exists for your deployment target: calibrate on a **small stratified** image subset (e.g. random COCO ids), apply **QAT** or **weight-only GPTQ** depending on stack, and re-run the same caption rubric; accept only if the share of worse captions on a fixed pilot set stays below team threshold (for example under 10%).