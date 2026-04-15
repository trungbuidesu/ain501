# Tier 1 Pipeline: router + parallel agents

Tier 1 extends the accessibility app (`src/app/main.py`) with:

1. **SLM router** (`src/router/slm_router.py`): Scene classifier ONNX INT8 → scene label + confidence; if confidence is below `router.confidence_threshold`, **fallback heuristics** apply (frame diff, edge density, skin-like pixels).
2. **Policy** `router.scene_to_agents`: maps each scene class to a set of experts: `object`, `action`, `ocr`, `face_pose`.
3. **Parallel runtime** (`src/agents/agent_manager.py`): `ThreadPoolExecutor` + **per-agent timeouts**; results merged with provenance (`status`, `latency_ms`).
4. **Experts**
   - `ObjectAgent`: YOLOv8n INT8 (existing).
   - `ActionAgent`: MoViNet-A0 INT8, clip `action_clip_frames` from a shared `RingBuffer`.
   - `OcrAgent`: PaddleOCR with coarse grid ROIs + optional full-frame fallback.
   - `FacePoseAgent`: MediaPipe Face + Pose Tasks (`.task` models cached under `reports/prebuilt_week5/_mediapipe_models/` by default).
5. **Captions**: `SpatialTemplateEngine` for objects; `src/caption/tier1_caption.py` appends action/OCR/face/pose utterances.

## Configuration (`configs/app.yaml`)

| Block | Purpose |
| --- | --- |
| `tier1.enabled` | Master switch (`false` keeps legacy behavior). |
| `tier1.strict_artifacts` | If `true`, missing ONNX for router/action raises at startup (recommended for delivery). |
| `tier1.object_only` | Debug: only YOLO, no router or other experts. CLI: `--object-only`. |
| `tier1.ring_buffer_maxlen` | Frames kept for temporal action clips. |
| `tier1.router` | `model_path`, `confidence_threshold`, `image_size`, `scene_to_agents`, `fallback` thresholds. |
| `tier1.timeouts` | Seconds per expert for `AgentManager`. |

## Run

```bash
python -m src.app --max-frames 200
```

Enable Tier 1 in YAML (`tier1.enabled: true`) after placing:

- `models/scene_classifier_int8.onnx`
- `models/movinet_a0_int8.onnx`
- `models/yolov8n_int8.onnx`
- Piper voice (see [app_runtime.md](app_runtime.md))

## Benchmarks

- **End-to-end**: `reports/app/app_latency.json` and `reports/phase3/tier1_benchmark.json` when `tier1.enabled` is true.
- **Per-agent CPU table**: `python -m src.training.scripts.benchmark_tier1_agents` → `reports/phase3/tier1_benchmark.md` (+ JSON sidecar).

Targets (same machine, CPU): action p50 &lt; 15 ms, OCR p50 &lt; 30 ms, face/pose p50 &lt; 15 ms — see report for pass/fail.

## Tests

- `tests/test_slm_router.py` — routing + fallback (mock ONNX).
- `tests/test_agent_manager.py` — parallelism + timeouts.
- `tests/test_action_agent.py` — mock session.
- `tests/test_phase3_integration.py` — stress + selective activation.
- `tests/test_face_pose_agent.py`, `tests/test_ocr_agent.py` — optional (MediaPipe / Paddle stack).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `FileNotFoundError` for scene or MoViNet ONNX | Train/export or copy artifacts; or set `tier1.strict_artifacts: false` only for local experiments. |
| Ultralow FPS | Reduce active experts in `scene_to_agents`, or raise `change_detection` SSIM threshold to emit fewer frames. |
| Paddle / MediaPipe errors | Install `paddlepaddle` CPU wheel + `paddleocr`; allow first-run download for `.task` models (network). |
