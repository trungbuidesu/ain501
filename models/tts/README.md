# Local TTS model layout (Piper)

Week 5 standardizes Piper voices under `models/tts/piper/`, mirroring the
[Hugging Face Piper voices](https://huggingface.co/rhasspy/piper-voices) tree.

If you previously used `models/voices/piper`, copy or symlink that directory to
`models/tts/piper` so paths in `configs/datasets/tts_piper_accessibility.yaml`
resolve.

Expected layout (example):

- `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx`
- `models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json`
- `models/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx`
- `models/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json`

Do not commit large ONNX weights unless your team policy allows it.
