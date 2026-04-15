# Scene Classifier (SLM Router)

Scene classifier is a lightweight router model for fast scene-state decisions.
It predicts 5 classes:

- `static`
- `motion`
- `text_present`
- `face_detected`
- `scene_change`

## Model Variants

- `tiny_cnn`: 3-4 conv blocks + FC head, target around 1M params.
- `mobilenet_head`: lightweight classifier head on top of MobileNetV3-Small features.

Config defaults are in `configs/models/scene_classifier.yaml`.

## Data Preparation

Unified manifest schema:

- `path`
- `label`
- `source`
- `split`

Class source mapping:

- `static`: Places365 subset
- `motion`: UCF-101 extracted frames
- `text_present`: TextOCR/ICDAR positives
- `face_detected`: LFW/WIDER Face subset
- `scene_change`: synthetic frame-pair transitions from UCF clips

Build manifest:

`python -m src.training.scripts.scene_classifier prepare-manifest --execute`

## Training

Train from scratch:

`python -m src.training.scripts.scene_classifier train --manifest-csv data/processed/scene_classifier/manifest.csv`

Main tracked metrics:

- `train/val loss`
- `train/val accuracy`
- per-class precision/recall/F1
- confusion matrix and misclassification CSV in checkpoint run directory

Optional quality gate:

- `--target-accuracy 0.85 --enforce-target-accuracy`

## QAT and Export

Run QAT fine-tuning:

`python -m src.training.scripts.scene_classifier qat --manifest-csv data/processed/scene_classifier/manifest.csv --checkpoint <best.pt> --output-checkpoint <qat.pt>`

Export ONNX FP32 + INT8:

`python -m src.training.scripts.scene_classifier export-onnx --checkpoint <best.pt> --verify-parity`

Expected deploy artifact:

- `models/scene_classifier_int8.onnx`

## Benchmark

Benchmark CPU latency:

`python -m src.training.scripts.scene_classifier benchmark --runtime onnxruntime --onnx-path models/scene_classifier_int8.onnx --output-json reports/scene_classifier/benchmark_current_cpu.json`

Target:

- `latency_ms_p50 < 5.0`
