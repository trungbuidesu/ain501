"""Kiến trúc MoViNet-A0 cho nhận diện hành động.

MoViNet (Mobile Video Networks) là dòng mô hình hiệu quả tối ưu cho thiết bị di động.
Kiến trúc này sử dụng causal convolutions và stream buffers để hỗ trợ xử lý video
online/streaming với độ trễ cực thấp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from movinets import MoViNet
from movinets.config import _C


class MoViNetA0Backbone(nn.Module):
    """Backbone MoViNet-A0 hỗ trợ stream buffer.

    Mô hình này được tinh chỉnh để loại bỏ head 600-class mặc định (Kinetics-600)
    và thay thế bằng head mới cho số lượng class yêu cầu.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_path: str | None = None,
        causal: bool = True,
        tf_like: bool = True,
    ) -> None:
        """Khởi tạo mô hình MoViNet-A0.

        Args:
            num_classes: Số lượng class đầu ra.
            pretrained_path: Đường dẫn tới file weights pretrained (.pt).
            causal: Nếu True, sử dụng causal convolutions (hỗ trợ stream buffer).
            tf_like: Nếu True, sử dụng padding giống TensorFlow.
        """
        super().__init__()

        # Xác định conv_type dựa trên causal (giống logic mặc định khi pretrained=True)
        conv_type = "2plus1d" if causal else "3d"

        # Khởi tạo mô hình gốc từ package movinets
        # Ta khởi tạo với num_classes=600 để khớp với pretrained weights ban đầu
        self.model = MoViNet(
            _C.MODEL.MoViNetA0,
            causal=causal,
            pretrained=False,
            tf_like=tf_like,
            num_classes=600,
            conv_type=conv_type,
        )

        # Load weights nếu được cung cấp
        if pretrained_path:
            state_dict = torch.load(
                pretrained_path, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state_dict)

        # Thay thế head phân loại sang số lượng class mới
        # Cấu trúc: classifier[3].conv_1.conv2d (2plus1d) hoặc .conv3d (3d)
        in_channels = 2048
        new_conv: nn.Module
        if conv_type == "2plus1d":
            new_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
            self.model.classifier[3].conv_1.conv2d = new_conv  # type: ignore
        else:
            new_conv = nn.Conv3d(in_channels, num_classes, kernel_size=(1, 1, 1))
            self.model.classifier[3].conv_1.conv3d = new_conv  # type: ignore

        # Đảm bảo head mới được khởi tạo ngẫu nhiên
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")  # type: ignore
        if hasattr(new_conv, "bias") and new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor đầu vào shape [B, C, T, H, W].

        Returns:
            Logits đầu ra shape [B, num_classes].
        """
        # MoViNet package trả về [B, num_classes] trực tiếp sau avgpool và classifier
        return self.model(x)

    def clean_activation_buffers(self) -> None:
        """Dọn dẹp stream buffer sau khi kết thúc một video.

        Cần được gọi trước khi bắt đầu xử lý clip đầu tiên của video mới
        khi sử dụng chế độ causal.
        """
        self.model.clean_activation_buffers()
