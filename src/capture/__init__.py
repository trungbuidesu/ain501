"""Multi-source video capture: unified RGB frames, ring buffer, paced FPS."""

from src.capture.fake_source import FakeFrameSource
from src.capture.paced_loop import pace_after_frame
from src.capture.protocol import FrameSource
from src.capture.ring_buffer import RingBuffer
from src.capture.types import FramePacket, ScreenRegion, SourceKind

__all__ = [
    "FakeFrameSource",
    "FramePacket",
    "FrameSource",
    "RingBuffer",
    "ScreenRegion",
    "SourceKind",
    "pace_after_frame",
]
