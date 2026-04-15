"""Sleep pacing to approximate a target capture FPS."""

from __future__ import annotations

import time


def pace_after_frame(iteration_start_mono: float, target_fps: float) -> None:
    """Sleep to align iteration duration with ``1/target_fps``.

    Call at the end of each capture iteration (after read + processing).
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    interval = 1.0 / target_fps
    elapsed = time.monotonic() - iteration_start_mono
    wait = interval - elapsed
    if wait > 0:
        time.sleep(wait)
