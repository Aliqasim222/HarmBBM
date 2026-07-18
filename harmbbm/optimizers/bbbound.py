"""Dynamic-bound Barzilai--Borwein optimizer."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch
from torch.optim import Optimizer


class BBbound(Optimizer):
    """Stochastic BB optimizer with a dynamic upper bound.

    This implementation follows the epochwise structure of the BBbound method:
    a gradient moving average is formed within each epoch, the BB candidate is
    computed from consecutive epoch-end parameter and moving-average snapshots,
    and the candidate is restricted by an upper bound derived from the previous
    accepted step.

    ``beta1`` controls the within-epoch gradient moving average. The published
    experimental setting uses ``beta1 = 4 / M``, where ``M`` is the number of
    mini-batches per epoch. ``beta2`` controls contraction of the dynamic upper
    bound.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.1,
        beta1: float = 0.01,
        beta2: float = 0.999,
        weight_decay: float = 5e-4,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 < beta1 <= 1.0:
            raise ValueError("beta1 must be in (0, 1].")
        if not 0.0 < beta2 < 1.0:
            raise ValueError("beta2 must be in (0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative.")
        if not 0.0 < min_lr <= max_lr:
            raise ValueError("Require 0 < min_lr <= max_lr.")
        if not min_lr <= lr <= max_lr:
            raise ValueError("Initial lr must lie in [min_lr, max_lr].")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        defaults = dict(
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            weight_decay=float(weight_decay),
            min_lr=float(min_lr),
            max_lr=float(max_lr),
            eps=float(eps),
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("BBbound currently supports one parameter group.")

        self._previous_parameters: Optional[list[torch.Tensor]] = None
        self._previous_gradient_average: Optional[list[torch.Tensor]] = None
        self._gradient_average = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in self.param_groups[0]["params"]
        ]
        self._observed_steps = 0

    @torch.no_grad()
    def begin_epoch(self) -> None:
        for average in self._gradient_average:
            average.zero_()
        self._observed_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1 = float(group["beta1"])
        weight_decay = float(group["weight_decay"])
        observed = False

        for index, parameter in enumerate(group["params"]):
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("BBbound does not support sparse gradients.")

            gradient = parameter.grad.detach()
            if weight_decay != 0.0:
                gradient = gradient.add(parameter, alpha=weight_decay)

            average = self._gradient_average[index]
            average.mul_(1.0 - beta1).add_(gradient, alpha=beta1)
            parameter.add_(gradient, alpha=-lr)
            observed = True

        if observed:
            self._observed_steps += 1
        return loss

    @torch.no_grad()
    def end_epoch(
        self,
        epoch_index: int,
        steps_per_epoch: int,
        decay_factor: float = 1.0,
    ) -> dict[str, Any]:
        """Update the bounded BB learning rate for the next epoch."""
        if epoch_index <= 0:
            raise ValueError("epoch_index must be positive and one-indexed.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if decay_factor <= 0:
            raise ValueError("decay_factor must be positive.")

        group = self.param_groups[0]
        eps = float(group["eps"])
        beta1 = float(group["beta1"])
        beta2 = float(group["beta2"])
        min_lr = float(group["min_lr"])
        max_lr = float(group["max_lr"])
        previous_lr = float(group["lr"])

        current_parameters = [
            parameter.detach().clone() for parameter in group["params"]
        ]
        current_average = [average.detach().clone() for average in self._gradient_average]

        s_dot_s = float("nan")
        s_dot_y = float("nan")
        raw_bb_lr = float("nan")
        dynamic_upper = float("nan")
        curvature_valid = False
        next_lr = previous_lr

        if (
            self._previous_parameters is not None
            and self._previous_gradient_average is not None
            and self._observed_steps > 0
        ):
            s_dot_s = 0.0
            s_dot_y = 0.0
            scale = float(steps_per_epoch)
            for current_p, previous_p, current_g, previous_g in zip(
                current_parameters,
                self._previous_parameters,
                current_average,
                self._previous_gradient_average,
            ):
                s_vector = ((current_p - previous_p) / scale).double()
                y_vector = (current_g - previous_g).double()
                s_dot_s += torch.sum(s_vector * s_vector).item()
                s_dot_y += torch.sum(s_vector * y_vector).item()

            if (
                math.isfinite(s_dot_s)
                and math.isfinite(s_dot_y)
                and s_dot_s > eps
                and abs(s_dot_y) > eps
            ):
                raw_bb_lr = s_dot_s / (abs(s_dot_y) + eps)

                base_increment = 1.0 / (
                    (1.0 - beta2) * (epoch_index * steps_per_epoch + 1.0)
                )
                exponent = beta1 * steps_per_epoch / float(epoch_index)
                log_upper = math.log(max(previous_lr, eps)) + exponent * math.log1p(
                    base_increment
                )
                if log_upper >= math.log(max_lr):
                    dynamic_upper = max_lr
                else:
                    dynamic_upper = math.exp(log_upper)

                candidate = min(raw_bb_lr, dynamic_upper)
                if math.isfinite(candidate) and candidate > 0.0:
                    next_lr = candidate
                    curvature_valid = True

        next_lr *= decay_factor
        next_lr = min(max(next_lr, min_lr), max_lr)
        group["lr"] = float(next_lr)

        self._previous_parameters = current_parameters
        self._previous_gradient_average = current_average
        self.begin_epoch()

        return {
            "used_lr": previous_lr,
            "next_lr": float(next_lr),
            "raw_bb_lr": raw_bb_lr,
            "dynamic_upper": dynamic_upper,
            "curvature_valid": curvature_valid,
            "s_dot_s": s_dot_s,
            "s_dot_y": s_dot_y,
            "decay_factor": float(decay_factor),
        }

    def current_lr(self) -> float:
        return float(self.param_groups[0]["lr"])
