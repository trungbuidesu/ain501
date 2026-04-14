"""Dataset và transform cho nhận diện hành động dùng MoViNet.

Module này hỗ trợ đọc video theo clip, áp dụng các kỹ thuật augmentation phổ biến
cho video như temporal jittering, random cropping và normalization.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.data.core import DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD


class VideoClipDataset(Dataset):
    """Dataset đọc video theo clip dành cho MoViNet.

    Đầu ra là tensor shape [C, T, H, W] chuẩn hóa. Tương thích với manifest CSV
    có các cột: path, label.
    """

    def __init__(
        self,
        manifest_path: Path,
        class_names: list[str],
        n_clip_frames: int = 6,
        frame_stride: int = 2,
        resolution: int = 172,
        is_train: bool = True,
        temporal_jitter: bool = True,
    ) -> None:
        """Khởi tạo VideoClipDataset.

        Args:
            manifest_path: Đường dẫn tới file CSV manifest.
            class_names: Danh sách tên các class (để map index).
            n_clip_frames: Số lượng frame trong mỗi clip.
            frame_stride: Khoảng cách giữa các frame.
            resolution: Kích thước ảnh (H=W).
            is_train: Nếu True, áp dụng augmentation (random crop, jitter).
            temporal_jitter: Nếu True, chọn điểm bắt đầu clip ngẫu nhiên.
        """
        self.n_clip_frames = n_clip_frames
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.is_train = is_train
        self.temporal_jitter = temporal_jitter

        # Đọc manifest
        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        # Map label sang index
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        # Normalization constants (CHW format) - Ensure float32 to avoid XPU errors
        self.mean = np.array(DEFAULT_NORMALIZE_MEAN, dtype=np.float32).reshape(
            3, 1, 1, 1
        )
        self.std = np.array(DEFAULT_NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Đọc và trả về một clip video cùng label index."""
        row = self.rows[index]
        video_path = row["path"]
        label_name = row["label"]
        label_idx = self.class_to_idx[label_name]

        # Mở video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tính toán span yêu cầu
        required_span = (self.n_clip_frames - 1) * self.frame_stride + 1

        # Chọn điểm bắt đầu (temporal jittering)
        if total_frames > required_span:
            if self.is_train and self.temporal_jitter:
                start_frame = np.random.randint(0, total_frames - required_span + 1)
            else:
                start_frame = (total_frames - required_span) // 2
        else:
            start_frame = 0

        # Seek tới điểm bắt đầu một lần duy nhất
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.n_clip_frames):
            ret, frame = cap.read()
            if not ret:
                # Fallback nếu không đọc được
                frame = (
                    frames[-1]
                    if frames
                    else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                )
            else:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self._process_frame(frame)
            frames.append(frame)

            # Skip các frames dựa trên stride (dùng grab sẽ nhanh hơn read/decode)
            for _ in range(self.frame_stride - 1):
                cap.grab()

        cap.release()

        # Stack thành [T, H, W, C] -> [C, T, H, W]
        video_np = np.stack(frames, axis=0).transpose(3, 0, 1, 2)  # [C, T, H, W]

        # Trả về uint8 để tiết kiệm VRAM, preprocessing sẽ làm trên GPU
        return torch.from_numpy(video_np), label_idx

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize và augment frame."""
        h, w = frame.shape[:2]

        # Resize sao cho cạnh ngắn nhất bằng resolution
        scale = self.resolution / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop hoặc Random crop
        if self.is_train:
            dy = np.random.randint(0, new_h - self.resolution + 1)
            dx = np.random.randint(0, new_w - self.resolution + 1)
        else:
            dy = (new_h - self.resolution) // 2
            dx = (new_w - self.resolution) // 2

        frame = frame[dy : dy + self.resolution, dx : dx + self.resolution]

        # Horizontal Flip ngẫu nhiên khi train
        if self.is_train and np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        return frame
