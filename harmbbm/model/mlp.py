"""MLP model used for the MNIST experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP1024(nn.Module):
    """Two-hidden-layer MLP: 784 -> 1024 -> 1024 -> num_classes."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
