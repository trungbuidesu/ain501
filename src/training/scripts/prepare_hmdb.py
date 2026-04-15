"""Prepare HMDB subset with stratified split and frame extraction."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

ZIP_PATH = Path("data/hmdb51.zip")
TARGET_CLASSES = [
    "walk",
    "wave",
    "sit",
    "stand",
    "climb_stairs",
    "fall_floor",
    "run",
    "turn",
    "push",
    "shake_hands",
    "talk",
    "clap",
]
OUTPUT_ROOT = Path("data/processed/action_accessibility")
VIDEOS_DIR = OUTPUT_ROOT / "videos"
FRAMES_DIR = OUTPUT_ROOT / "frames"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
FRAMES_MANIFEST_PATH = OUTPUT_ROOT / "manifest_frames.csv"

MAX_FRAMES_PER_VIDEO = 300
TARGET_FPS = 5
SEED = 501


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for HMDB preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=ZIP_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_VIDEO)
    parser.add_argument("--target-fps", type=int, default=TARGET_FPS)
    parser.add_argument("--force", action="store_true")
    return parser


def extract_subset(zip_path: Path, videos_dir: Path) -> None:
    """Extract only target class videos from hmdb zip archive."""

    videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting target classes from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in tqdm(archive.namelist()):
            parts = Path(member).parts
            if len(parts) < 2:
                continue
            class_name = parts[-2]
            if class_name not in TARGET_CLASSES or not member.endswith(".avi"):
                continue
            target_file = videos_dir / class_name / parts[-1]
            if target_file.exists():
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_file.open("wb") as target:
                target.write(source.read())


def generate_manifest(
    videos_dir: Path,
    manifest_path: Path,
    train_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    """Generate deterministic stratified manifest with 80/20 custom split."""

    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    print("Generating stratified custom split...")
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for class_name in TARGET_CLASSES:
        class_dir = videos_dir / class_name
        videos = sorted(class_dir.glob("*.avi"))
        rng.shuffle(videos)
        if not videos:
            continue
        split_idx = int(len(videos) * train_ratio)
        split_idx = max(1, min(len(videos) - 1, split_idx)) if len(videos) > 1 else 1
        for index, video_path in enumerate(videos):
            split = "train" if index < split_idx else "test"
            relative = video_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            rows.append(
                {
                    "path": relative,
                    "label": class_name,
                    "split": split,
                    "group": video_path.stem,
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    *,
    target_fps: int,
) -> bool:
    """Extract frames using ffmpeg with 5-digit naming."""

    output_pattern = output_dir / "frame_%05d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={target_fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def extract_frames_cv2(video_path: Path, output_dir: Path, max_frames: int) -> int:
    """Fallback extraction with OpenCV and 5-digit frame names."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frame_name = f"frame_{count:05d}.jpg"
        cv2.imwrite(
            str(output_dir / frame_name),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        count += 1
    cap.release()
    return count


def process_all_frames(
    manifest_rows: list[dict[str, str]],
    *,
    frames_dir: Path,
    max_frames: int,
    target_fps: int,
) -> list[dict[str, str]]:
    """Extract frames and write rows compatible with FrameClipDataset."""

    print("Extracting frames...")
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    for row in tqdm(manifest_rows):
        video_path = Path(row["path"])
        label = row["label"]
        split = row["split"]
        group = row["group"]
        output_subdir = frames_dir / label / group
        output_subdir.mkdir(parents=True, exist_ok=True)
        has_ffmpeg = extract_frames_ffmpeg(
            video_path,
            output_subdir,
            target_fps=target_fps,
        )
        if not has_ffmpeg:
            _ = extract_frames_cv2(video_path, output_subdir, max_frames=max_frames)
        frames = sorted(output_subdir.glob("frame_*.jpg"))
        if max_frames > 0 and len(frames) > max_frames:
            for path in frames[max_frames:]:
                path.unlink(missing_ok=True)
            frames = frames[:max_frames]
        if not frames:
            continue
        relative_frame_dir = output_subdir.resolve().relative_to(Path.cwd().resolve())
        output_rows.append(
            {
                "frame_dir": relative_frame_dir.as_posix(),
                "label": label,
                "split": split,
                "n_frames": str(len(frames)),
            }
        )
    return output_rows


def write_frames_manifest(rows: list[dict[str, str]], path: Path) -> None:
    """Write frame manifest for FrameClipDataset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_dir", "label", "split", "n_frames"])
        writer.writeheader()
        writer.writerows(rows)


def print_split_report(rows: list[dict[str, str]]) -> None:
    """Print compact split and class report."""

    by_split = Counter(row["split"] for row in rows)
    by_class_split: dict[str, Counter[str]] = {}
    for row in rows:
        by_class_split.setdefault(row["label"], Counter())
        by_class_split[row["label"]][row["split"]] += 1
    print(f"rows={len(rows)} split_counts={dict(by_split)}")
    for class_name in sorted(by_class_split):
        train_count = by_class_split[class_name].get("train", 0)
        test_count = by_class_split[class_name].get("test", 0)
        total = train_count + test_count
        ratio = train_count / total if total else 0.0
        print(
            f"class={class_name} train={train_count} test={test_count} "
            f"train_ratio={ratio:.3f}"
        )


def main() -> None:
    """Run HMDB preparation workflow."""

    args = build_arg_parser().parse_args()
    output_root = args.output_root
    videos_dir = output_root / "videos"
    frames_dir = output_root / "frames"
    manifest_path = output_root / "manifest.csv"
    frames_manifest_path = output_root / "manifest_frames.csv"

    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    if not args.zip_path.exists():
        raise FileNotFoundError(f"Missing zip archive: {args.zip_path}")

    extract_subset(args.zip_path, videos_dir)
    manifest_rows = generate_manifest(
        videos_dir=videos_dir,
        manifest_path=manifest_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    frame_rows = process_all_frames(
        manifest_rows,
        frames_dir=frames_dir,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
    )
    write_frames_manifest(frame_rows, frames_manifest_path)
    print_split_report(frame_rows)
    print(f"manifest={manifest_path}")
    print(f"frames_manifest={frames_manifest_path}")


if __name__ == "__main__":
    main()
