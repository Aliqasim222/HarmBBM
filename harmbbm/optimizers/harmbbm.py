"""Harmonic Barzilai--Borwein Momentum optimizer."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch
from torch.optim import Optimizer


class HarmBBM(Optimizer):
    """Harmonic Barzilai--Borwein Momentum.

    The method uses a single global learning rate during each epoch. At the
    epoch boundary, it forms an epoch-level secant pair and computes:

    * a Barzilai--Borwein spectral candidate;
    * secant agreement between the displacement and gradient change;
    * average positive gradient--momentum agreement during the epoch.

    The two agreement terms define a confidence weight. The next learning rate
    is the weighted harmonic mean of the BB candidate and the scheduled SGDM
    learning rate. Invalid curvature gives zero confidence and an exact fallback
    to the scheduled SGDM rate.

    The mini-batch momentum convention matches ``torch.optim.SGD`` without
    dampening:

        v_t = beta * v_{t-1} + g_t
        theta_{t+1} = theta_t - eta_k * v_t
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        q_max: float = 0.9,
        min_lr: float = 1e-6,
        max_lr: float = 2e-1,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative.")
        if not 0.0 <= q_max < 1.0:
            raise ValueError("q_max must be in [0, 1).")
        if not 0.0 < min_lr <= max_lr:
            raise ValueError("Require 0 < min_lr <= max_lr.")
        if not min_lr <= lr <= max_lr:
            raise ValueError("Initial lr must lie in [min_lr, max_lr].")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            q_max=float(q_max),
            min_lr=float(min_lr),
            max_lr=float(max_lr),
            eps=float(eps),
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("HarmBBM currently supports one parameter group.")

        parameters = list(self.param_groups[0]["params"])
        self._previous_parameters: Optional[list[torch.Tensor]] = None
        self._previous_average_gradients: Optional[list[torch.Tensor]] = None
        self._gradient_sums = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in parameters
        ]
        self._gradient_count = 0
        self._agreement_sum = 0.0
        self._agreement_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        weight_decay = float(group["weight_decay"])
        eps = float(group["eps"])

        dot_gm = 0.0
        norm_g_sq = 0.0
        norm_m_sq = 0.0
        has_previous_momentum = False
        effective_gradients: list[tuple[int, torch.nn.Parameter, torch.Tensor]] = []

        for index, parameter in enumerate(group["params"]):
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("HarmBBM does not support sparse gradients.")

            gradient = parameter.grad.detach()
            if weight_decay != 0.0:
                gradient = gradient.add(parameter, alpha=weight_decay)
            effective_gradients.append((index, parameter, gradient))
            self._gradient_sums[index].add_(gradient)

            buffer = self.state[parameter].get("momentum_buffer")
            if buffer is not None:
                has_previous_momentum = True
                gradient64 = gradient.double()
                buffer64 = buffer.double()
                dot_gm += torch.sum(gradient64 * buffer64).item()
                norm_g_sq += torch.sum(gradient64 * gradient64).item()
                norm_m_sq += torch.sum(buffer64 * buffer64).item()

        if effective_gradients:
            self._gradient_count += 1
            if has_previous_momentum and norm_g_sq > eps and norm_m_sq > eps:
                cosine = dot_gm / (math.sqrt(norm_g_sq * norm_m_sq) + eps)
                self._agreement_sum += min(1.0, max(0.0, cosine))
                self._agreement_count += 1

        for _, parameter, gradient in effective_gradients:
            state = self.state[parameter]
            if momentum != 0.0:
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = gradient.clone().detach()
                    state["momentum_buffer"] = buffer
                else:
                    buffer.mul_(momentum).add_(gradient)
                direction = buffer
            else:
                direction = gradient
            parameter.add_(direction, alpha=-lr)

        return loss

    @torch.no_grad()
    def end_epoch(self, scheduled_lr_next: float) -> dict[str, Any]:
        """Compute and install the learning rate for the next epoch."""
        if scheduled_lr_next <= 0:
            raise ValueError("scheduled_lr_next must be positive.")

        group = self.param_groups[0]
        eps = float(group["eps"])
        q_max = float(group["q_max"])
        min_lr = float(group["min_lr"])
        max_lr = float(group["max_lr"])
        used_lr = float(group["lr"])
        scheduled_lr_next = min(max(float(scheduled_lr_next), min_lr), max_lr)

        parameters = [parameter.detach().clone() for parameter in group["params"]]
        if self._gradient_count > 0:
            average_gradients = [
                gradient_sum.detach().clone().div_(self._gradient_count)
                for gradient_sum in self._gradient_sums
            ]
        else:
            average_gradients = [torch.zeros_like(parameter) for parameter in group["params"]]

        gradient_momentum_agreement = (
            self._agreement_sum / self._agreement_count
            if self._agreement_count > 0
            else 0.0
        )

        s_dot_s = float("nan")
        s_dot_y = float("nan")
        y_dot_y = float("nan")
        secant_agreement = float("nan")
        bb_lr = float("nan")
        confidence = 0.0
        curvature_valid = False
        next_lr = scheduled_lr_next

        if (
            self._previous_parameters is not None
            and self._previous_average_gradients is not None
            and self._gradient_count > 0
        ):
            s_dot_s = 0.0
            s_dot_y = 0.0
            y_dot_y = 0.0
            for current_p, previous_p, current_g, previous_g in zip(
                parameters,
                self._previous_parameters,
                average_gradients,
                self._previous_average_gradients,
            ):
                s_vector = (current_p - previous_p).double()
                y_vector = (current_g - previous_g).double()
                s_dot_s += torch.sum(s_vector * s_vector).item()
                s_dot_y += torch.sum(s_vector * y_vector).item()
                y_dot_y += torch.sum(y_vector * y_vector).item()

            finite = all(math.isfinite(v) for v in (s_dot_s, s_dot_y, y_dot_y))
            if finite and s_dot_s > eps and s_dot_y > eps and y_dot_y > eps:
                bb_lr = s_dot_s / (s_dot_y + eps)
                secant_agreement = s_dot_y / (
                    math.sqrt(s_dot_s * y_dot_y) + eps
                )
                secant_agreement = min(1.0, max(0.0, secant_agreement))

                if math.isfinite(bb_lr) and bb_lr > 0.0:
                    confidence = min(
                        q_max,
                        secant_agreement * gradient_momentum_agreement,
                    )
                    denominator = (
                        (1.0 - confidence) / scheduled_lr_next
                        + confidence / bb_lr
                    )
                    if math.isfinite(denominator) and denominator > eps:
                        candidate = 1.0 / denominator
                        if math.isfinite(candidate) and candidate > 0.0:
                            next_lr = candidate
                            curvature_valid = True

        if not curvature_valid:
            confidence = 0.0
            next_lr = scheduled_lr_next

        next_lr = min(max(next_lr, min_lr), max_lr)
        group["lr"] = float(next_lr)

        self._previous_parameters = parameters
        self._previous_average_gradients = average_gradients
        for gradient_sum in self._gradient_sums:
            gradient_sum.zero_()
        self._gradient_count = 0
        self._agreement_sum = 0.0
        self._agreement_count = 0

        return {
            "used_lr": used_lr,
            "scheduled_lr_next": scheduled_lr_next,
            "next_lr": float(next_lr),
            "bb_lr": bb_lr,
            "secant_agreement": secant_agreement,
            "gradient_momentum_agreement": gradient_momentum_agreement,
            "agreement_confidence": confidence,
            "curvature_valid": curvature_valid,
            "s_dot_s": s_dot_s,
            "s_dot_y": s_dot_y,
            "y_dot_y": y_dot_y,
        }

    def current_lr(self) -> float:
        return float(self.param_groups[0]["lr"])
