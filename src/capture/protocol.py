"""Frame source protocol for tests and live capture."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.capture.types import FramePacket


@runtime_checkable
class FrameSource(Protocol):
    """Readable capture source; implementations return RGB :class:`FramePacket`."""

    def read(self) -> FramePacket | None:
        """Return next frame or ``None`` when exhausted / unavailable."""

    def close(self) -> None:
        """Release cameras, files, or screen handles."""
