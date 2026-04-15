"""Ghi frame màn hình phục vụ bộ dữ liệu tùy chỉnh (accessibility).

Chạy qua ``python -m src.training.scripts.capture_frames``. Mặc định **không**
ghi file (dry-run): chỉ tính số frame dự kiến. Truyền ``--execute`` để lưu JPEG.
Ưu tiên thư viện ``dxcam`` (Windows / DirectX); nếu không có thì dùng ``mss``
chụp toàn màn hình chính.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import mss

from src.capture.screen_backend import create_dxcam_if_available, grab_screen_rgb


def build_parser() -> argparse.ArgumentParser:
    """Dựng CLI: ``--output-dir``, ``--duration-seconds``, ``--fps``, ``--execute``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/custom/raw_frames"),
    )
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def capture_frames(
    output_dir: Path,
    duration_seconds: int,
    fps: float,
    execute: bool = False,
) -> int:
    """Lập kế hoạch hoặc thực sự chụp frame màn hình theo chu kỳ ``1/fps``.

    Args:
        output_dir: Thư mục đích cho ``frame_*.jpg`` khi ``execute`` là True.
        duration_seconds: Tổng thời gian ghi (giây), phải dương.
        fps: Số frame trên giây (float dương).
        execute: False chỉ trả về số frame dự kiến; True tạo thư mục và ghi ảnh.

    Returns:
        Số frame dự kiến (dry-run) hoặc số frame đã ghi thành công (có thể nhỏ
        hơn nếu ``dxcam.grab()`` trả về None).

    Raises:
        ValueError: Nếu ``duration_seconds`` hoặc ``fps`` không dương.
    """

    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration_seconds and fps must be positive")
    frame_count = int(duration_seconds * fps)
    if not execute:
        return frame_count

    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / fps
    camera = create_dxcam_if_available()
    with mss.mss() as sct:
        return _capture_screen_loop(
            output_dir, frame_count, interval, camera, sct
        )


def _capture_screen_loop(
    output_dir: Path,
    frame_count: int,
    interval: float,
    dxcam_camera: object | None,
    mss_instance: object,
) -> int:
    """Ghi frame màn hình (dxcam hoặc mss) và lưu JPEG; dùng chung ``screen_backend``."""
    from PIL import Image

    captured = 0
    for index in range(frame_count):
        rgb = grab_screen_rgb(
            dxcam_camera=dxcam_camera,
            mss_instance=mss_instance,
            region=None,
        )
        if rgb is not None:
            Image.fromarray(rgb).save(output_dir / f"frame_{index:06d}.jpg")
            captured += 1
        time.sleep(interval)
    return captured


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, gọi ``capture_frames``, in số frame và luôn trả ``0``."""
    args = build_parser().parse_args(argv)
    count = capture_frames(
        output_dir=args.output_dir,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        execute=args.execute,
    )
    print(f"frames={'captured' if args.execute else 'planned'} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
