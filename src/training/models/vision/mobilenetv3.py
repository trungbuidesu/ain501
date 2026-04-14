# -*- coding: utf-8 -*-
"""MobileNetV3-Small shared feature extractor.

MobileNetV3-Small is built from inverted residual blocks, uses
squeeze-and-excitation in selected blocks, and relies on hard-swish/hard-sigmoid
activations for mobile-friendly nonlinearities. This module removes the
torchvision classification head and keeps the convolutional feature extractor,
average pooling, and flattened 576-dimensional embedding output.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def _resolve_weights(weights: str | None) -> MobileNet_V3_Small_Weights | None:
    """Map CLI/config weight aliases to torchvision weights objects."""

    if weights is None or weights.lower() in {"none", "random"}:
        return None
    if weights.lower() in {"default", "imagenet"}:
        return MobileNet_V3_Small_Weights.DEFAULT
    raise ValueError("weights must be one of: default, imagenet, none")


def build_mobilenetv3_small_backbone(
    weights: str | None = "default",
) -> MobileNetV3SmallBackbone:
    """Build a MobileNetV3-Small backbone with the requested weights."""

    return MobileNetV3SmallBackbone(weights=weights)


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small without the classification head.

    The kept path is `features -> avgpool -> flatten`. The removed torchvision
    head is the MLP classifier that maps the 576-dimensional pooled embedding to
    ImageNet logits. Downstream agents can use `forward_feature_map` for spatial
    maps or `forward_embedding`/`forward` for compact vectors.
    """

    embedding_dim = 576

    def __init__(self, weights: str | None = "default") -> None:
        """Load torchvision MobileNetV3-Small and drop its classifier."""

        super().__init__()
        model = mobilenet_v3_small(weights=_resolve_weights(weights))
        self.features = model.features
        self.avgpool = model.avgpool

    def forward_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the final convolutional feature map, normally `[N,576,7,7]`."""

        return cast(torch.Tensor, self.features(inputs))

    def forward_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the pooled 576-dimensional embedding for each image."""

        feature_map = self.forward_feature_map(inputs)
        pooled = self.avgpool(feature_map)
        return torch.flatten(pooled, start_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias for `forward_embedding` so export defaults to embeddings."""

        return self.forward_embedding(inputs)


class MobileNetV3SmallWithHead(nn.Module):
    """Temporary classifier wrapper used only for supervised fine-tuning."""

    def __init__(
        self,
        backbone: MobileNetV3SmallBackbone,
        num_classes: int,
    ) -> None:
        """Attach a linear classifier to a MobileNetV3-Small backbone."""

        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits from backbone embeddings."""

        return cast(
            torch.Tensor, self.classifier(self.backbone.forward_embedding(inputs))
        )


def freeze_mobilenetv3_early_layers(
    model: MobileNetV3SmallBackbone | MobileNetV3SmallWithHead,
    freeze_until: int = 7,
) -> None:
    """Freeze early MobileNetV3 feature blocks and leave later blocks trainable."""

    backbone = model.backbone if isinstance(model, MobileNetV3SmallWithHead) else model
    for index, layer in enumerate(backbone.features):
        requires_grad = index >= freeze_until
        for parameter in layer.parameters():
            parameter.requires_grad = requires_grad
    if isinstance(model, MobileNetV3SmallWithHead):
        _set_requires_grad(model.classifier.parameters(), True)


def _set_requires_grad(parameters: Iterable[nn.Parameter], value: bool) -> None:
    """Set `requires_grad` on an iterable of parameters."""

    for parameter in parameters:
        parameter.requires_grad = value
