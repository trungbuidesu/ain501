"""Ring buffer of the most recent :class:`FramePacket` instances."""

from __future__ import annotations

from collections import deque

from src.capture.types import FramePacket


class RingBuffer:
    """Fixed-capacity deque of :class:`FramePacket`; oldest dropped when full."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._dq: deque[FramePacket] = deque(maxlen=maxlen)

    def push(self, packet: FramePacket) -> None:
        self._dq.append(packet)

    def __len__(self) -> int:
        return len(self._dq)

    def __iter__(self):
        return iter(self._dq)

    def latest(self) -> FramePacket | None:
        return self._dq[-1] if self._dq else None

    def as_list(self) -> list[FramePacket]:
        """Return a snapshot of buffered packets in chronological order."""

        return list(self._dq)

    def clear(self) -> None:
        self._dq.clear()
