"""ResNet models used for the PathMNIST experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, in_channels: int = 3) -> nn.Module:
    """Build a ResNet18 adapted to 28x28 images.

    The ImageNet stem is replaced by a 3x3, stride-one convolution and the
    initial max-pooling operation is removed.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
