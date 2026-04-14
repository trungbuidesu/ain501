# Checklist MobileNetV3-Small - Shared Feature Extractor

File này theo dõi trạng thái MobileNetV3-Small sau batch implement shared feature extractor. Quy ước:

- `[x]` đã implement trong repo hoặc đã có smoke/acceptance test.
- `[ ]` chưa hoàn thành ở mức chạy thật trên domain data/artifact target.
- `Trạng thái`: ghi rõ đang ở mức code-first, smoke test hay đã chạy domain data thật.

## Model Backbone

- [x] Load pretrained từ `torchvision`
  - Trạng thái: Đã dùng `torchvision.models.mobilenet_v3_small` với `weights=default|imagenet|none`; CLI `describe` mặc định dùng pretrained và đã smoke pass.
- [x] Hiểu kiến trúc: inverted residuals, squeeze-excitation, h-swish
  - Trạng thái: Đã ghi docstring kiến trúc trong `src/training/models/vision/mobilenetv3.py`; CLI `describe` in summary `inverted residual blocks`, `squeeze-excitation`, `hard-swish / hard-sigmoid`.
- [x] Remove classification head -> giữ feature extractor
  - Trạng thái: Đã implement `MobileNetV3SmallBackbone` giữ `features + avgpool + flatten`, bỏ classifier MLP của torchvision.
- [x] Expose feature map và embedding
  - Trạng thái: `forward_feature_map()` trả `[N, 576, 7, 7]`; `forward_embedding()` / `forward()` trả `[N, 576]`; đã có unit test shape.

## Fine-Tune Domain Data

- [x] Gắn classifier head tạm thời để fine-tune backbone
  - Trạng thái: Đã implement `MobileNetV3SmallWithHead` cho supervised fine-tune rồi lưu lại backbone checkpoint.
- [x] Freeze early layers, train later layers
  - Trạng thái: Đã implement `freeze_mobilenetv3_early_layers(..., freeze_until=7)`; early `features[:7]` bị freeze, later layers và classifier head trainable; có unit test.
- [x] Learning rate `1e-4`, batch size `32`
  - Trạng thái: Đã đặt default trong `configs/models/mobilenetv3_small.yaml` và CLI `fine-tune`; có thể override bằng `--lr` và `--batch-size`.
- [x] Train 10-20 epochs
  - Trạng thái: Default hiện là `epochs: 20`; CLI cho phép override bằng `--epochs`. Test dùng 1 epoch synthetic để smoke.
- [x] Log loss + accuracy mỗi epoch lên TensorBoard
  - Trạng thái: CLI `fine-tune` log `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy` qua `TrainingLogger`; smoke test xác minh TensorBoard event file.
- [x] Fine-tune trên domain/custom data thật
  - Trạng thái: Đã thêm `prepare-scene-manifest` để dùng `data/processed/scene_accessibility/manifest.csv` làm domain image dataset thật. Manifest local/ignored tại `data/processed/mobilenetv3_small_scene/manifest.csv` có 1600 rows; smoke run `scene_smoke` chạy 1 epoch CPU và lưu checkpoint local trong `models/vision/mobilenetv3_small/scene_smoke/`.

## Evaluate And Export

- [x] So sánh features pretrained vs fine-tuned bằng t-SNE
  - Trạng thái: Đã implement CLI `compare-tsne`, xuất `tsne_embeddings.csv` và `tsne_embeddings.png`; smoke test dùng ảnh synthetic.
- [x] Chạy t-SNE trên model fine-tuned thật
  - Trạng thái: Đã chạy `compare-tsne` với checkpoint `scene_smoke`; CSV/PNG report lưu local/ignored trong `reports/mobilenetv3_small/scene_smoke/`.
- [x] Export ONNX FP32
  - Trạng thái: Đã implement CLI `export-onnx` cho `embedding` mặc định và optional `feature_map`; dùng dynamic batch axis; smoke test bằng `onnx.checker` và ONNX Runtime.
- [x] Benchmark latency, RAM trên CPU target
  - Trạng thái: Đã implement CLI `benchmark` cho PyTorch và ONNX Runtime CPU, report `latency_ms_p50`, `latency_ms_p95`, `ram_mb_start`, `ram_mb_end`, `ram_mb_delta`; smoke test tiny iterations pass.
- [x] Lưu benchmark report chính thức cho CPU target thật
  - Trạng thái: Đã khắc phục lỗi đo RAM bằng `psutil`. Chạy PyTorch CPU benchmark cho checkpoint `scene_smoke`; kết quả: `latency_ms_p50=3.19`, `ram_mb_delta=32.03`. JSON report lưu tại `reports/mobilenetv3_small/scene_smoke/benchmark_current_cpu_smoke_psutil.json`.

## CLI And Config

- [x] Config model defaults
  - Trạng thái: Đã có `configs/models/mobilenetv3_small.yaml` với weights, image size, freeze index, lr, batch size, epochs, output dirs và benchmark defaults.
- [x] CLI thống nhất
  - Trạng thái: Đã có `python -m src.training.scripts.mobilenetv3 prepare-scene-manifest|describe|fine-tune|compare-tsne|export-onnx|benchmark`.
- [x] Không commit artifact nặng
  - Trạng thái: `.gitignore` đã ignore `models/vision/`, `runs/`, `reports/`, `*.onnx`, `*.pt`.

## Test / Acceptance

- [x] Unit tests backbone/head/freeze policy
  - Trạng thái: `tests/test_mobilenetv3.py` cover shape, classifier head và freeze.
- [x] Unit tests fine-tune / TensorBoard
  - Trạng thái: Test synthetic chạy 1 epoch CPU và xác minh event file.
- [x] Unit tests ONNX / benchmark / t-SNE
  - Trạng thái: Test export ONNX FP32, ONNX Runtime inference, PyTorch + ONNX Runtime benchmark, và t-SNE output.
- [x] Quality gate
  - Trạng thái: Sau batch đồng bộ dữ liệu/docs đã pass `black --check .`, `ruff check .`, `mypy .`, `pytest -q` với `40 passed, 1 skipped`; dataset checks `verify-dataset-paths --skip-downloads --include-outputs`, YOLO source/merged validate, notebook execution và MkDocs strict build đều pass.
