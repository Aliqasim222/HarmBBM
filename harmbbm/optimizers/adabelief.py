"""Minimal PyTorch implementation of the standard AdaBelief optimizer."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief with coupled L2 weight decay.

    The implementation uses the residual between the observed gradient and the
    exponential first moment as the second-moment signal.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        beta1, beta2 = betas
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("betas must lie in [0, 1).")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative.")

        defaults = dict(
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients.")

                gradient = parameter.grad.detach()
                if weight_decay != 0.0:
                    gradient = gradient.add(parameter, alpha=weight_decay)

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_var"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]

                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                residual = gradient - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    residual,
                    residual,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / max(
                    bias_correction1,
                    1e-16,
                )
                denominator = exp_avg_var.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss
