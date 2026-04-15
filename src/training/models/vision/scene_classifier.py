"""Scene classifier architectures for routing labels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

from src.training.models.vision.mobilenetv3 import MobileNetV3SmallBackbone

SCENE_ROUTER_CLASSES: tuple[str, ...] = (
    "static",
    "motion",
    "text_present",
    "face_detected",
    "scene_change",
)


class TinySceneCNN(nn.Module):
    """Small CNN classifier with about one million parameters."""

    def __init__(
        self,
        num_classes: int = len(SCENE_ROUTER_CLASSES),
        channels: Sequence[int] = (32, 64, 128, 256),
        fc_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(channels) < 3:
            raise ValueError("TinySceneCNN requires at least 3 conv layers")
        if fc_hidden_dim <= 0:
            raise ValueError("fc_hidden_dim must be positive")

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return spatial features before the classifier."""

        return self.features(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for routing classes."""

        features = self.forward_features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


class MobileNetSceneHead(nn.Module):
    """Lightweight classifier head over MobileNet embeddings."""

    def __init__(
        self,
        num_classes: int = len(SCENE_ROUTER_CLASSES),
        weights: str | None = "default",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.backbone = MobileNetV3SmallBackbone(weights=weights)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embedding_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding from MobileNet backbone."""

        return self.backbone.forward_embedding(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits."""

        embedding = self.forward_embedding(inputs)
        return self.classifier(embedding)


def build_scene_classifier(
    variant: str,
    num_classes: int = len(SCENE_ROUTER_CLASSES),
    weights: str | None = "default",
) -> nn.Module:
    """Build a scene classifier model by variant name."""

    variant_name = variant.lower().strip()
    if variant_name == "tiny_cnn":
        return TinySceneCNN(num_classes=num_classes)
    if variant_name == "mobilenet_head":
        return MobileNetSceneHead(num_classes=num_classes, weights=weights)
    raise ValueError("variant must be one of: tiny_cnn, mobilenet_head")


def freeze_mobilenet_scene_backbone(model: MobileNetSceneHead, freeze: bool = True) -> None:
    """Freeze or unfreeze MobileNet backbone parameters."""

    _set_requires_grad(model.backbone.parameters(), not freeze)


def _set_requires_grad(parameters: Iterable[nn.Parameter], value: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = value
