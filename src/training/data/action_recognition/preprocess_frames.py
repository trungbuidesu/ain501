"""Script tiền xử lý: giải mã video thành JPEG frames (chạy 1 lần).

Thay vì để training loop giải mã video liên tục trên CPU, script này giải mã
toàn bộ dataset 1 lần, lưu frames dưới dạng JPEG. Khi train, dataset chỉ cần
đọc file JPEG từ SSD — cực nhẹ cho CPU, GPU luôn được nạp đầy dữ liệu.

Cách dùng:
    python -m src.training.data.action_recognition.preprocess_frames \\
        --manifest data/processed/action_accessibility/manifest.csv \\
        --output  data/processed/action_accessibility/frames \\
        --out-manifest data/processed/action_accessibility/manifest_frames.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Ảnh JPEG quality — 90 là điểm cân bằng tốt giữa chất lượng và kích thước
JPEG_QUALITY = 90
# Resize cạnh ngắn nhất về kích thước này để tiết kiệm I/O khi đọc
SHORT_SIDE = 256


def _decode_one(args: tuple[str, str, str]) -> tuple[str, str, int]:
    """Giải mã một video và lưu frames thành JPEG.

    Args:
        args: (video_path, frame_dir, group) — được truyền qua multiprocessing.

    Returns:
        Tuple (frame_dir, label, n_frames) để ghi vào manifest mới.
    """
    video_path, frame_dir_str, label = args
    frame_dir = Path(frame_dir_str)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    n_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize cạnh ngắn nhất về SHORT_SIDE (giữ tỷ lệ)
        h, w = frame.shape[:2]
        if min(h, w) != SHORT_SIDE:
            scale = SHORT_SIDE / min(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        out_path = frame_dir / f"frame_{n_frames:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n_frames += 1

    cap.release()
    return frame_dir_str, label, n_frames


def main(argv: list[str] | None = None) -> int:
    """Entry point cho script tiền xử lý."""
    parser = argparse.ArgumentParser(
        description="Giải mã video UCF-101 thành JPEG frames"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/action_accessibility/manifest.csv"),
        help="Đường dẫn manifest CSV gốc (có cột path, label, split)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/action_accessibility/frames"),
        help="Thư mục gốc lưu frames",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("data/processed/action_accessibility/manifest_frames.csv"),
        help="Manifest mới cho FrameClipDataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, mp.cpu_count()),
        help="Số tiến trình song song (mặc định: min(4, cpu_count))",
    )
    args = parser.parse_args(argv)

    # Đọc manifest gốc
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Tổng số video cần giải mã: {len(rows)}")

    # Chuẩn bị danh sách task
    tasks: list[tuple[str, str, str]] = []
    for row in rows:
        video_path = row["path"]
        label = row["label"]
        group = row["group"]
        frame_dir = str(args.output / label / group)
        tasks.append((video_path, frame_dir, label))

    # Giải mã song song
    results: list[tuple[str, str, int]] = []
    with mp.Pool(args.workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_decode_one, tasks),
            total=len(tasks),
            desc="Decoding",
        ):
            results.append(res)

    # Đếm frame thực tế (một số video có thể fail)
    ok = [r for r in results if r[2] > 0]
    print(f"Hoàn thành: {len(ok)}/{len(tasks)} video")

    # Ghi manifest mới: frame_dir, label, split, n_frames
    # Dùng lại split từ manifest gốc
    group_to_split = {row["group"]: row["split"] for row in rows}

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_dir", "label", "split", "n_frames"]
        )
        writer.writeheader()
        for res in results:
            frame_dir_str, label, n_frames = res
            if n_frames == 0:
                continue
            # Lấy group từ đường dẫn
            group = Path(frame_dir_str).name
            split = group_to_split.get(group, "train")
            writer.writerow(
                {
                    "frame_dir": frame_dir_str,
                    "label": label,
                    "split": split,
                    "n_frames": n_frames,
                }
            )

    print(f"Manifest mới: {args.out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
