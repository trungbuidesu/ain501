"""Phase 1 capture: unified frame format, ring buffer, sources, change detection."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2
import mss
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

import src.capture.app as capture_app
from src.capture.app import run_capture_app
from src.capture.config import ChangeDetectionConfig, load_capture_config
from src.capture.fake_source import FakeFrameSource
from src.capture.paced_loop import pace_after_frame
from src.capture.ring_buffer import RingBuffer
from src.capture.sources import FileVideoSource, WebcamSource, frame_source_from_config
from src.capture.types import FramePacket, ScreenRegion
from src.detection.change_detector import ChangeDetector


def test_frame_packet_rgb_uint8() -> None:
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    p = FramePacket(data=data, t_mono=0.0, source="webcam", frame_id=0)
    assert p.data.shape == (4, 4, 3)
    assert p.data.dtype == np.uint8


def test_frame_packet_rejects_bad_shape() -> None:
    bad = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        FramePacket(data=bad, t_mono=0.0, source="webcam", frame_id=0)


def test_fake_source_100_frames_uniform_format() -> None:
    src = FakeFrameSource(64, 48, 100, source="webcam", pattern="solid")
    n = 0
    while True:
        pkt = src.read()
        if pkt is None:
            break
        n += 1
        assert pkt.data.dtype == np.uint8
        assert pkt.data.shape == (64, 48, 3)
        assert pkt.source == "webcam"
        assert pkt.frame_id == n - 1
    assert n == 100
    src.close()


def test_ring_buffer_maxlen() -> None:
    rb = RingBuffer(3)
    for i in range(5):
        d = np.full((2, 2, 3), i, dtype=np.uint8)
        rb.push(
            FramePacket(data=d, t_mono=0.0, source="webcam", frame_id=i),
        )
    assert len(rb) == 3
    assert rb.latest() is not None
    assert int(rb.latest().data[0, 0, 0]) == 4


def test_file_source_100_frames(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32))
    assert w.isOpened()
    for _ in range(100):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()

    src = FileVideoSource(path, loop=False)
    count = 0
    while True:
        pkt = src.read()
        if pkt is None:
            break
        count += 1
        assert pkt.data.dtype == np.uint8
        assert pkt.data.shape == (32, 32, 3)
        assert pkt.source == "file"
    src.close()
    assert count == 100


@pytest.mark.skip(reason="requires physical webcam")
def test_webcam_100_frames() -> None:
    src = WebcamSource(0)
    for _ in range(100):
        pkt = src.read()
        assert pkt is not None
        assert pkt.data.dtype == np.uint8
        assert pkt.data.ndim == 3 and pkt.data.shape[2] == 3
    src.close()


@pytest.mark.skip(reason="requires screen capture stack (CI/headless)")
def test_screen_100_frames() -> None:
    from src.capture.sources import ScreenSource

    src = ScreenSource(None)
    for _ in range(100):
        pkt = src.read()
        assert pkt is not None
        assert pkt.data.dtype == np.uint8
    src.close()


def test_pace_after_frame_non_negative_sleep() -> None:
    t0 = time.monotonic()
    pace_after_frame(t0, target_fps=1000.0)
    assert time.monotonic() >= t0


def test_change_detector_identical_skips_after_first() -> None:
    cfg = ChangeDetectionConfig(
        method="ssim",
        ssim_trigger_below=0.92,
        changed_fraction_trigger_above=0.02,
        pixel_diff_threshold=25,
        max_metric_side=64,
    )
    det = ChangeDetector(cfg)
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    assert det.should_process(None, a) is True
    assert det.should_process(a, a.copy()) is False


def test_change_detector_different_triggers() -> None:
    cfg = ChangeDetectionConfig(
        method="pixel_diff",
        ssim_trigger_below=0.99,
        changed_fraction_trigger_above=0.01,
        pixel_diff_threshold=5,
        max_metric_side=64,
    )
    det = ChangeDetector(cfg)
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert det.should_process(None, a) is True
    assert det.should_process(a, b) is True


def test_load_capture_config_sample() -> None:
    cfg = load_capture_config(
        Path(__file__).resolve().parents[1] / "configs" / "capture_pipeline.yaml",
    )
    assert cfg.source.type in ("webcam", "screen", "file")
    assert cfg.timing.target_fps > 0


def test_frame_source_from_config_webcam() -> None:
    cfg = load_capture_config(
        Path(__file__).resolve().parents[1] / "configs" / "capture_pipeline.yaml",
    )
    src = frame_source_from_config(cfg)
    src.close()


def test_frame_source_from_config_screen_honors_backend(monkeypatch: MonkeyPatch) -> None:
    cfg = load_capture_config(
        Path(__file__).resolve().parents[1] / "configs" / "capture_pipeline.yaml",
    )
    cfg = replace(
        cfg,
        source=replace(cfg.source, type="screen"),
        screen=replace(cfg.screen, backend="mss"),
    )
    seen: dict[str, str | None] = {"backend": None}

    class _DummyScreenSource:
        def __init__(self, region, *, backend: str = "auto") -> None:
            seen["backend"] = backend
            self._region = region

        def close(self) -> None:
            return None

    monkeypatch.setattr("src.capture.sources.ScreenSource", _DummyScreenSource)
    src = frame_source_from_config(cfg)
    src.close()
    assert seen["backend"] == "mss"


def test_integration_fake_capture_counts_changes() -> None:
    cfg = ChangeDetectionConfig(
        method="ssim",
        ssim_trigger_below=0.95,
        changed_fraction_trigger_above=0.05,
        pixel_diff_threshold=25,
        max_metric_side=96,
    )
    det = ChangeDetector(cfg)
    src = FakeFrameSource(16, 16, 10, pattern="solid")
    prev = None
    changes = 0
    skips = 0
    while True:
        pkt = src.read()
        if pkt is None:
            break
        if det.should_process(prev, pkt.data):
            changes += 1
            prev = pkt.data.copy()
        else:
            skips += 1
    src.close()
    assert changes == 1
    assert skips == 9


@pytest.mark.slow
def test_memory_loop_short_fake() -> None:
    import psutil

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    src = FakeFrameSource(32, 32, 500, pattern="solid")
    buf = RingBuffer(30)
    while True:
        pkt = src.read()
        if pkt is None:
            break
        buf.push(pkt)
    src.close()
    rss1 = proc.memory_info().rss
    growth_mb = (rss1 - rss0) / (1024 * 1024)
    assert growth_mb < 200.0


def test_run_capture_app_file_source(tmp_path: Path) -> None:
    path = tmp_path / "short.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w = cv2.VideoWriter(str(path), fourcc, 10.0, (16, 16))
    for _ in range(30):
        w.write(np.zeros((16, 16, 3), dtype=np.uint8))
    w.release()

    cfg = load_capture_config(
        Path(__file__).resolve().parents[1] / "configs" / "capture_pipeline.yaml",
    )
    cfg = replace(
        cfg,
        source=replace(
            cfg.source,
            type="file",
            file_path=path,
            file_loop=False,
        ),
        timing=replace(cfg.timing, target_fps=60.0),
    )
    assert run_capture_app(cfg, max_frames=20) == 0


def test_run_capture_app_respects_max_frames(monkeypatch: MonkeyPatch) -> None:
    cfg = load_capture_config(
        Path(__file__).resolve().parents[1] / "configs" / "capture_pipeline.yaml",
    )
    pkt = FramePacket(
        data=np.zeros((8, 8, 3), dtype=np.uint8),
        t_mono=0.0,
        source="webcam",
        frame_id=0,
    )

    class _DummySource:
        def __init__(self) -> None:
            self.read_calls = 0
            self.closed = False

        def read(self):
            self.read_calls += 1
            if self.read_calls <= 3:
                return pkt
            return None

        def close(self) -> None:
            self.closed = True

    dummy = _DummySource()

    class _AlwaysChange:
        def __init__(self, _cfg) -> None:
            return None

        def should_process(self, _prev, _curr) -> bool:
            return True

    monkeypatch.setattr(capture_app, "frame_source_from_config", lambda _cfg: dummy)
    monkeypatch.setattr(capture_app, "ChangeDetector", _AlwaysChange)
    assert run_capture_app(cfg, max_frames=1) == 0
    assert dummy.read_calls >= 1
    assert dummy.closed is True


def test_screen_region_crop_integration() -> None:
    from src.capture.screen_backend import grab_rgb_mss

    r = ScreenRegion(left=0, top=0, width=10, height=10)
    with mss.mss() as sct:
        img = grab_rgb_mss(sct, r)
    assert img.shape == (10, 10, 3)
    assert img.dtype == np.uint8
