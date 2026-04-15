# Active learning (MVP)

Minimal loop for adding hard examples after deployment, without a heavy UI.

## Layout

| Path | Role |
| --- | --- |
| `data/active_learning/inbox/` | New frames plus `*.meta.json` sidecars from capture |
| `data/active_learning/annotations/` | Human CSV edits (path, corrected label) |
| `data/active_learning/merged/` | Manifest snippets ready to merge into training data |

## 1. Capture

Use screen capture as usual (`python -m src.training.scripts.capture_frames --execute`) or copy candidate images from your detector. Then register them with metadata:

```bash
python -m src.training.scripts.active_learning_capture ^
  --image path\to\frame.jpg ^
  --model-id scene_classifier ^
  --confidence 0.42 ^
  --prediction-json "{\"top1\":\"motion\"}"
```

Optional: `--prediction-file preds.json` for larger payloads. Each run copies the image into `data/active_learning/inbox/` and writes `<timestamp>_<stem>.meta.json` next to it.

## 2. Annotate

Create `data/active_learning/annotations/batch1.csv`:

```csv
path,label_correct,notes
data/active_learning/inbox/123.456_frame.jpg,static_face,
```

Extra columns `split` and `source` are optional (defaults: `train`, `active_learning`).

## 3. Merge into scene manifest

```bash
python -m src.training.scripts.active_learning_merge ^
  --annotations data/active_learning/annotations/batch1.csv ^
  --output-csv data/active_learning/merged/scene_append.csv
```

Concatenate the result with your existing scene manifest (or replace paths after moving files into `data/processed/...`), then re-run training and `benchmark_suite` / registry updates.

## Other tasks (YOLO, video)

This MVP targets image + CSV flows that match `scene_classifier` manifests. Object-detection box labels need a separate tool or editor; keep those exports in `data/active_learning/annotations/` with a format your YOLO pipeline expects, then merge manually.

## Hygiene

- Remove reviewed items from `inbox/` after merging to avoid duplicate labels.
- Keep `notes` for class confusion patterns; they help the next training log.
