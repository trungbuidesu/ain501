from __future__ import annotations

import numpy as np

from src.capture.types import FramePacket
from src.encoder.preprocess import preprocess_mobilenet_frame
from src.encoder.visual_encoder import VisualEncoder


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEncoderSession:
    def get_inputs(self):
        return [_FakeInput("images")]

    def run(self, _outs, feeds):
        x = feeds["images"]
        batch = x.shape[0]
        # Return fixed embedding size for deterministic tests.
        return [np.ones((batch, 576), dtype=np.float32)]


def test_preprocess_mobilenet_shape_and_dtype() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    out = preprocess_mobilenet_frame(frame, image_size=224)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_visual_encoder_cache_hits() -> None:
    session = _FakeEncoderSession()
    enc = VisualEncoder(session=session, enable_cache=True, cache_size=8)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    key = ("webcam", 1, None)
    v1 = enc.encode(frame, cache_key=key)
    v2 = enc.encode(frame, cache_key=key)
    assert v1.shape == (576,)
    assert v2.shape == (576,)
    assert enc.cache_stats["hits"] >= 1


def test_visual_encoder_encode_packet() -> None:
    session = _FakeEncoderSession()
    enc = VisualEncoder(session=session, enable_cache=True)
    pkt = FramePacket(
        data=np.zeros((32, 32, 3), dtype=np.uint8),
        t_mono=0.0,
        source="webcam",
        frame_id=7,
    )
    vec = enc.encode_packet(pkt)
    assert vec.shape == (576,)


def test_visual_encoder_latency_verify_missing_report(tmp_path) -> None:
    session = _FakeEncoderSession()
    enc = VisualEncoder(session=session, enable_cache=False)
    result = enc.verify_latency_against_phase0(
        1.0,
        benchmark_report_path=tmp_path / "missing.json",
    )
    assert result["status"] == "inconclusive"
