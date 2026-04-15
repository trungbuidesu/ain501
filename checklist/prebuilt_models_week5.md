# Checklist 0.9 Week 5 — Pre-built models (verify only)

Convention: `[x]` done, `[ ]` pending. Update **Trạng thái** after you run on hardware.

### Trạng thái PaddleOCR

Trạng thái: not run / in progress / done — note Paddle + PaddleOCR versions and report paths.

### Trạng thái MediaPipe

Trạng thái: not run / in progress / done — note MediaPipe wheel version.

### Trạng thái TTS

Trạng thái: voices missing / `validate-tts` ok / benchmark done — note `models/tts/piper` layout.

## PaddleOCR lite

- [ ] Install PaddlePaddle (CPU) + PaddleOCR; record versions in `reports/prebuilt_week5/paddleocr_benchmark_cpu.json`.
- [ ] Select lite model (e.g. mobile det/rec); record name and source URL in docs.
- [ ] Prepare 20 images with visible text (or use `--synthetic` for smoke).
- [ ] Run `python -m src.training.scripts.verify_prebuilt_week5 paddleocr ...`; output `reports/prebuilt_week5/paddleocr_samples.jsonl`.
- [ ] CPU benchmark: p50/p95 per image in JSON.
- [ ] Update `docs/models/paddleocr.md` with output schema and example row.

## MediaPipe Face Mesh + Pose

- [ ] Pin MediaPipe version in report.
- [ ] Face Mesh: 20 face images; export landmarks; optional heuristic **emotion proxy** (rule-based, not FER).
- [ ] Pose: 20 images/frames; keypoints + rule-based pose label (stand/sit/arms_up).
- [ ] CPU benchmark: `reports/prebuilt_week5/mediapipe_benchmark_cpu.json`.
- [ ] Update `docs/models/mediapipe.md` with landmark indices, coords, pose thresholds.

## TTS

- [ ] Fill `reports/prebuilt_week5/tts_compare.md` (Piper vs Kokoro vs pyttsx3).
- [ ] Place Piper vi+en ONNX + json under `models/tts/piper/...` (see `configs/datasets/tts_piper_accessibility.yaml`).
- [ ] `python -m src.training.scripts.data_utils validate-tts --execute` (or strict dry-run if models missing).
- [ ] 20 sample sentences -> WAV under `data/tts_validation/week5/` + subjective notes.
- [ ] Benchmark text-to-first-audio -> `reports/prebuilt_week5/piper_benchmark_cpu.json`.
- [ ] Interrupt test: long utterance vs short alert (document app-level stop behavior).
- [ ] Update `docs/models/piper_tts.md` with paths and I/O.

## Week 5 closeout

- [ ] All artifacts under `reports/prebuilt_week5/`.
- [ ] No large binaries committed (check `.gitignore`).
- [ ] `pytest` / docs build if applicable.
