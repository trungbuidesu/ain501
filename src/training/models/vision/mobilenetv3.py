# -*- coding: utf-8 -*-
"""Shared feature extractor MobileNetV3-Small.

MobileNetV3-Small dùng inverted residual blocks, squeeze-excitation ở một số
block và hard-swish/hard-sigmoid activations. Module này bỏ classification
head mặc định, giữ convolutional extractor + avgpool để xuất embedding 576
chiều.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def _resolve_weights(weights: str | None) -> MobileNet_V3_Small_Weights | None:
    """Resolve weight aliases sang torchvision weight objects."""

    if weights is None or weights.lower() in {"none", "random"}:
        return None
    if weights.lower() in {"default", "imagenet"}:
        return MobileNet_V3_Small_Weights.DEFAULT
    raise ValueError("weights must be one of: default, imagenet, none")


def build_mobilenetv3_small_backbone(
    weights: str | None = "default",
) -> MobileNetV3SmallBackbone:
    """Tạo backbone MobileNetV3-Small với weights được yêu cầu."""

    return MobileNetV3SmallBackbone(weights=weights)


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small backbone không chứa classification head.

    Luồng forward giữ `features -> avgpool -> flatten`. Classifier MLP của
    torchvision, vốn ánh xạ embedding 576 chiều sang ImageNet logits, bị loại
    bỏ. Downstream có thể gọi `forward_feature_map` để lấy tensor không gian
    hoặc `forward_embedding`/`forward` để lấy compact vectors.
    """

    embedding_dim = 576

    def __init__(self, weights: str | None = "default") -> None:
        """Load torchvision model gốc và bỏ classifier head."""

        super().__init__()
        model = mobilenet_v3_small(weights=_resolve_weights(weights))
        self.features = model.features
        self.avgpool = model.avgpool

    def forward_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        """Trả final convolutional feature map, thường là `[N,576,7,7]`."""

        return cast(torch.Tensor, self.features(inputs))

    def forward_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Trả pooled embedding 576 chiều cho từng ảnh."""

        feature_map = self.forward_feature_map(inputs)
        pooled = self.avgpool(feature_map)
        return torch.flatten(pooled, start_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias cho `forward_embedding`."""

        return self.forward_embedding(inputs)


class MobileNetV3SmallWithHead(nn.Module):
    """Wrapper classifier tạm thời cho supervised fine-tuning."""

    def __init__(
        self,
        backbone: MobileNetV3SmallBackbone,
        num_classes: int,
    ) -> None:
        """Gắn linear classifier lên MobileNetV3-Small backbone."""

        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Trả class logits từ backbone embeddings."""

        return cast(
            torch.Tensor, self.classifier(self.backbone.forward_embedding(inputs))
        )


def freeze_mobilenetv3_early_layers(
    model: MobileNetV3SmallBackbone | MobileNetV3SmallWithHead,
    freeze_until: int = 7,
) -> None:
    """Freeze early MobileNetV3 blocks và để later blocks trainable."""

    backbone = model.backbone if isinstance(model, MobileNetV3SmallWithHead) else model
    for index, layer in enumerate(backbone.features):
        requires_grad = index >= freeze_until
        for parameter in layer.parameters():
            parameter.requires_grad = requires_grad
    if isinstance(model, MobileNetV3SmallWithHead):
        _set_requires_grad(model.classifier.parameters(), True)


def _set_requires_grad(parameters: Iterable[nn.Parameter], value: bool) -> None:
    """Set `requires_grad` cho một iterable parameters."""

    for parameter in parameters:
        parameter.requires_grad = value
