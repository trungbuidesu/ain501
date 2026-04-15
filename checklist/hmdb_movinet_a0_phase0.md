# Checklist: HMDB-51 subset + MoViNet-A0 (Phase 0)

Tracks the HMDB-51 (12-class) retrain of MoViNet-A0: stratified **80/20** train/test, **30** epochs, **16** frames / **stride 2**, **Intel Arc A770 (XPU)**.

Convention:

- `[x]` Done (code + real run with artifacts where noted).
- `[/]` In progress.
- `[ ]` Not done or not verified on target hardware.
- **Status**: smoke, pilot, or production.

## Goals

- 12 HMDB classes: `configs/datasets/action_accessibility.yaml` -> `classes.hmdb51_accessibility`.
- Split: **80% train / 20% test**, per class, seed **501**.
- Training: **30 epochs**, **n_clip_frames: 16**, **frame_stride: 2**, **resolution: 172** (`configs/models/movinet_a0.yaml`).
- Device: **XPU**.

## Data preparation

- [x] Extract subset from `data/hmdb51.zip` to `data/processed/action_accessibility/videos/`
  - Status: `src/training/scripts/prepare_hmdb.py` (`python -m src.training.scripts.prepare_hmdb`).
- [x] Write video manifest with split + group
  - Status: columns `path,label,split,group` -> `data/processed/action_accessibility/manifest.csv`.
- [x] Decode frames compatible with `FrameClipDataset`
  - Status: `frame_00000.jpg`; columns `frame_dir,label,split,n_frames` -> `data/processed/action_accessibility/manifest_frames.csv`.
- [x] Every class has both train and test
  - Status: 2,196 clips; 1,752 train / 444 test; all 12 classes covered.

## Configuration

- [x] `configs/datasets/action_accessibility.yaml` — HMDB paths, 12 classes, `split_seed: 501`, `split_strategy: stratified_80_20_custom`
  - Status: aligned for Phase 0.
- [x] `configs/models/movinet_a0.yaml` — `num_classes: 12`, `epochs: 30`, `n_clip_frames: 16`, `frame_stride: 2`, `training.device: xpu`
  - Status: aligned for this batch.

## Training and evaluation

- [x] XPU dry-run (one batch)
  - Status: `movinet fine-tune --dry-run`; logits shape `[16, 12]`.
- [x] Full fine-tune (30 epochs) on XPU
  - Status: completed; early-epoch loss trend in `reports/movinet_a0/hmdb_retrain_summary.json`.
- [x] Evaluate test split (Top-1)
  - Status: **0.5203**; checkpoint `models/video/movinet_a0/best.pt`.
- [x] Error analysis (top confused pairs)
  - Status: e.g. `walk->run (20)`, `turn->walk (16)`, `run->walk (11)` via `movinet error-analysis`.

## Artifacts and reports

- [x] Checkpoint: `models/video/movinet_a0/best.pt`
- [x] Summary JSON: `reports/movinet_a0/hmdb_retrain_summary.json`
- [ ] Optional: ONNX export / PTQ for HMDB checkpoint
  - Status: not required for Phase 0; use `movinet export-onnx` / `ptq` when deploying.

## Reference commands

```text
python -m src.training.scripts.prepare_hmdb --zip-path data/hmdb51.zip --output-root data/processed/action_accessibility --train-ratio 0.8 --seed 501

python -m src.training.scripts.movinet --config configs/models/movinet_a0.yaml --manifest data/processed/action_accessibility/manifest.csv --frames-manifest data/processed/action_accessibility/manifest_frames.csv fine-tune --dry-run

python -m src.training.scripts.movinet --config configs/models/movinet_a0.yaml --manifest data/processed/action_accessibility/manifest.csv --frames-manifest data/processed/action_accessibility/manifest_frames.csv fine-tune

python -m src.training.scripts.movinet --config configs/models/movinet_a0.yaml --manifest data/processed/action_accessibility/manifest.csv --frames-manifest data/processed/action_accessibility/manifest_frames.csv evaluate

python -m src.training.scripts.movinet --config configs/models/movinet_a0.yaml --manifest data/processed/action_accessibility/manifest.csv --frames-manifest data/processed/action_accessibility/manifest_frames.csv error-analysis
```

## Notes

- Disk: extracted videos + frames may use on the order of **5–10 GB** depending on subset and per-video frame cap.
- Parent MoViNet checklist (includes HMDB batch section): `checklist/movinet_a0.md`.
