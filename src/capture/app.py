"""Phase 1 capture loop: config, optional region pick, FPS log, change/skip counts."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from src.capture.config import CapturePipelineConfig, load_capture_config
from src.capture.paced_loop import pace_after_frame
from src.capture.region_selector import select_screen_region_interactive
from src.capture.ring_buffer import RingBuffer
from src.capture.sources import frame_source_from_config
from src.detection.change_detector import ChangeDetector


def _resolve_config_path(raw_path: Path) -> Path:
    """Resolve config path robustly across cwd differences and trailing-dot typos."""

    candidates: list[Path] = [raw_path]
    text = str(raw_path)
    if text.endswith("."):
        candidates.append(Path(text.rstrip(".")))

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if not candidate.is_absolute():
            c1 = Path.cwd() / candidate
            if c1.exists():
                return c1
            c2 = repo_root / candidate
            if c2.exists():
                return c2
    # Fallback for error reporting.
    c = candidates[-1]
    return c if c.is_absolute() else (Path.cwd() / c)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/capture_pipeline.yaml"),
        help="Path to capture_pipeline YAML",
    )
    p.add_argument(
        "--select-region",
        action="store_true",
        help="Open drag overlay to pick screen region (screen source only)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (default: run until source ends)",
    )
    return p


def run_capture_app(cfg: CapturePipelineConfig, *, max_frames: int | None) -> int:
    src = frame_source_from_config(cfg)
    det = ChangeDetector(cfg.change_detection)
    buf = RingBuffer(cfg.timing.buffer_size)
    prev_rgb = None
    n_change = 0
    n_skip = 0
    intervals: deque[float] = deque(maxlen=60)

    print(
        "Capture loop started (Ctrl+C to stop). Logging frames...",
        file=sys.stderr,
        flush=True,
    )
    n_read = 0
    try:
        while True:
            t_iter = time.monotonic()
            pkt = src.read()
            if pkt is None:
                print("source ended or read failed", file=sys.stderr)
                break
            n_read += 1
            if max_frames is not None and n_read >= max_frames:
                break
            buf.push(pkt)
            emit = det.should_process(prev_rgb, pkt.data)
            if emit:
                n_change += 1
                prev_rgb = pkt.data.copy()
            else:
                n_skip += 1
            pace_after_frame(t_iter, cfg.timing.target_fps)
            elapsed = time.monotonic() - t_iter
            intervals.append(elapsed)
            avg_dt = sum(intervals) / len(intervals)
            inst_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            if pkt.frame_id % 15 == 0 or pkt.frame_id < 3:
                print(
                    f"frame={pkt.frame_id} emit={emit} inst_fps~={inst_fps:.2f} "
                    f"changes={n_change} skips={n_skip}",
                    flush=True,
                )
    finally:
        src.close()

    print(f"done changes={n_change} skips={n_skip}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    resolved_config = _resolve_config_path(args.config)
    if not resolved_config.exists():
        print(
            "config file not found. "
            f"arg={args.config} resolved={resolved_config} cwd={Path.cwd()}",
            file=sys.stderr,
        )
        return 2
    cfg = load_capture_config(resolved_config)

    want_region = args.select_region or cfg.screen.select_region_on_start
    if cfg.source.type == "screen" and want_region:
        region = select_screen_region_interactive()
        if region is None:
            print("region selection cancelled", file=sys.stderr)
            return 1
        cfg = replace(cfg, screen=replace(cfg.screen, region=region))

    return run_capture_app(cfg, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
