"""Optimizer factory for the five methods used in the paper."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .adabelief import AdaBelief
from .bbbound import BBbound
from .harmbbm import HarmBBM

SUPPORTED_OPTIMIZERS = ("sgdm", "adam", "adabelief", "bbbound", "harmbbm")


def build_optimizer(
    name: str,
    model: nn.Module,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> Optimizer:
    name = name.lower().strip()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: {', '.join(SUPPORTED_OPTIMIZERS)}"
        )

    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))

    if name == "sgdm":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(
                float(config.get("beta1", 0.9)),
                float(config.get("beta2", 0.999)),
            ),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    if name == "adabelief":
        return AdaBelief(
            model.parameters(),
            lr=lr,
            betas=(
                float(config.get("beta1", 0.9)),
                float(config.get("beta2", 0.999)),
            ),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    if name == "bbbound":
        beta1_value = config.get("beta1", "auto")
        if isinstance(beta1_value, str):
            if beta1_value.lower() != "auto":
                raise ValueError("BBbound beta1 must be numeric or 'auto'.")
            beta1 = min(1.0, 4.0 / max(1, steps_per_epoch))
        else:
            beta1 = float(beta1_value)
        return BBbound(
            model.parameters(),
            lr=lr,
            beta1=beta1,
            beta2=float(config.get("beta2", 0.999)),
            weight_decay=weight_decay,
            min_lr=float(config.get("min_lr", 1e-8)),
            max_lr=float(config.get("max_lr", 1.0)),
            eps=float(config.get("eps", 1e-12)),
        )
    return HarmBBM(
        model.parameters(),
        lr=lr,
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=weight_decay,
        q_max=float(config.get("q_max", 0.9)),
        min_lr=float(config.get("min_lr", 1e-6)),
        max_lr=float(config.get("max_lr", 0.2)),
        eps=float(config.get("eps", 1e-12)),
    )
