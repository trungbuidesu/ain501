"""Dataset đọc từ frames JPEG đã được giải mã sẵn.

Thay thế VideoClipDataset (đọc video trực tiếp bằng OpenCV — CPU nặng),
FrameClipDataset chỉ đọc file JPEG từ ổ đĩa, cực nhẹ cho CPU.
Toàn bộ preprocessing (normalization, float conversion) được làm trên GPU.

Manifest CSV cần có các cột:
    - frame_dir: đường dẫn thư mục chứa JPEG frames
    - label: tên class
    - split: "train" hoặc "test"
    - n_frames: tổng số frames trong thư mục
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FrameClipDataset(Dataset):
    """Dataset đọc clip từ thư mục JPEG frames đã giải mã sẵn.

    So với VideoClipDataset (giải mã video trực tiếp), FrameClipDataset
    không giải mã video — chỉ đọc JPEG, CPU sử dụng gần bằng 0, GPU
    được nạp đầy liên tục.

    Đầu ra: tensor uint8 [C, T, H, W] — preprocessing sẽ làm trên GPU.
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
        """Khởi tạo FrameClipDataset.

        Args:
            manifest_path: Đường dẫn tới file CSV manifest_frames.csv.
            class_names: Danh sách tên class (để map sang index).
            n_clip_frames: Số frame trong mỗi clip.
            frame_stride: Bước nhảy giữa các frame.
            resolution: Kích thước crop vuông (H=W).
            is_train: Nếu True, áp dụng random crop + flip.
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

        # Map label -> index
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self) -> int:
        """Số lượng mẫu trong dataset."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Đọc và trả về một clip cùng label index.

        Chỉ đọc JPEG từ đĩa — không giải mã video, CPU cực nhẹ.
        Trả về uint8 tensor để preprocessing được làm trên GPU.
        """
        row = self.rows[index]
        frame_dir = Path(row["frame_dir"])
        label_name = row["label"]
        label_idx = self.class_to_idx[label_name]
        n_frames = int(row["n_frames"])

        # Tính span cần thiết
        required_span = (self.n_clip_frames - 1) * self.frame_stride + 1

        # Chọn điểm bắt đầu (temporal jittering)
        if n_frames > required_span:
            if self.is_train and self.temporal_jitter:
                start = np.random.randint(0, n_frames - required_span + 1)
            else:
                start = (n_frames - required_span) // 2
        else:
            start = 0

        frames = []
        for i in range(self.n_clip_frames):
            idx = min(start + i * self.frame_stride, n_frames - 1)
            frame_path = frame_dir / f"frame_{idx:05d}.jpg"

            if frame_path.exists():
                # imread trả về BGR uint8
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    frame = self._blank_frame()
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self._process_frame(frame)
            else:
                frame = self._blank_frame()

            frames.append(frame)

        # Stack [T, H, W, C] -> [C, T, H, W] uint8
        video_np = np.stack(frames, axis=0).transpose(3, 0, 1, 2)
        # Trả về uint8: tiết kiệm 4x RAM và 4x băng thông RAM->VRAM
        return torch.from_numpy(video_np), label_idx

    def _blank_frame(self) -> np.ndarray:
        """Trả về frame đen khi không đọc được file."""
        return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize và augment một frame RGB.

        Args:
            frame: Mảng HWC uint8 định dạng RGB.

        Returns:
            Frame đã được resize và crop về (resolution x resolution).
        """
        h, w = frame.shape[:2]

        # Resize cạnh ngắn nhất về resolution (giữ tỷ lệ)
        scale = self.resolution / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h != h or new_w != w:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop hoặc Random crop
        if self.is_train:
            max_dy = max(0, new_h - self.resolution)
            max_dx = max(0, new_w - self.resolution)
            dy = np.random.randint(0, max_dy + 1)
            dx = np.random.randint(0, max_dx + 1)
        else:
            dy = (new_h - self.resolution) // 2
            dx = (new_w - self.resolution) // 2

        frame = frame[dy : dy + self.resolution, dx : dx + self.resolution]

        # Horizontal flip ngẫu nhiên khi train
        if self.is_train and np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        return frame
