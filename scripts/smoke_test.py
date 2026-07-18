#!/usr/bin/env python3
"""Run fast synthetic checks without downloading datasets."""

from __future__ import annotations

import torch
import torch.nn as nn

from harmbbm.optimizers import BBbound, HarmBBM


def _one_step(model: nn.Module, optimizer, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    torch.manual_seed(42)
    inputs = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = HarmBBM(model.parameters(), lr=0.01, max_lr=0.2)
    for _ in range(2):
        for _ in range(3):
            _one_step(model, optimizer, inputs, targets)
        optimizer.end_epoch(0.01)
    assert 0.0 < optimizer.current_lr() <= 0.2

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = BBbound(
        model.parameters(),
        lr=0.1,
        beta1=1.0,
        beta2=0.999,
        max_lr=1.0,
    )
    for epoch in range(1, 3):
        optimizer.begin_epoch()
        for _ in range(3):
            _one_step(model, optimizer, inputs, targets)
        optimizer.end_epoch(epoch_index=epoch, steps_per_epoch=3)
    assert 0.0 < optimizer.current_lr() <= 1.0

    print("HarmBBM and BBbound synthetic smoke tests passed.")


if __name__ == "__main__":
    main()
