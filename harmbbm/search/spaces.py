"""Deterministic hyperparameter candidate generation."""

from __future__ import annotations

import itertools
import random
from typing import Any


def _deduplicate(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            unique.append(config)
            seen.add(key)
    return unique


def _full_grid(name: str, dataset: str) -> list[dict[str, Any]]:
    dataset = dataset.lower()
    name = name.lower()
    wd_values = (
        [1e-5, 1e-4, 5e-4]
        if dataset == "mnist"
        else [1e-4, 5e-4, 1e-3]
    )

    if name == "sgdm":
        return [
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "weight_decay": float(weight_decay),
            }
            for lr, momentum, nesterov, weight_decay in itertools.product(
                [0.1, 0.05, 0.02, 0.01, 0.005],
                [0.8, 0.9, 0.95],
                [False, True],
                wd_values,
            )
        ]

    if name in {"adam", "adabelief"}:
        return [
            {
                "lr": float(lr),
                "beta1": float(beta1),
                "beta2": float(beta2),
                "eps": 1e-8,
                "weight_decay": float(weight_decay),
            }
            for lr, beta1, beta2, weight_decay in itertools.product(
                [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                [0.85, 0.9, 0.95],
                [0.99, 0.999],
                wd_values,
            )
        ]

    if name == "bbbound":
        return [
            {
                "lr": float(lr),
                "beta1": "auto",
                "beta2": float(beta2),
                "weight_decay": float(weight_decay),
                "min_lr": 1e-8,
                "max_lr": float(max_lr),
                "eps": 1e-12,
            }
            for lr, beta2, max_lr, weight_decay in itertools.product(
                [0.2, 0.1, 0.05, 0.01],
                [0.99, 0.995, 0.999],
                [0.2, 0.5, 1.0],
                wd_values,
            )
            if lr <= max_lr
        ]

    if name == "harmbbm":
        return [
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "q_max": float(q_max),
                "weight_decay": float(weight_decay),
                "min_lr": 1e-6,
                "max_lr": float(max_lr),
                "eps": 1e-12,
            }
            for lr, momentum, q_max, max_lr, weight_decay in itertools.product(
                [0.1, 0.05, 0.01, 0.005],
                [0.9, 0.95],
                [0.7, 0.9, 0.95],
                [0.05, 0.1, 0.2],
                wd_values,
            )
            if lr <= max_lr
        ]

    raise ValueError(f"No search space is defined for optimizer '{name}'.")


def build_search_candidates(
    optimizer_name: str,
    dataset: str,
    default_config: dict[str, Any],
    budget: int,
    tune_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic random-budget subset with the default first."""
    if budget <= 0:
        raise ValueError("Search budget must be positive.")

    default = dict(default_config)
    candidates = _deduplicate([default] + _full_grid(optimizer_name, dataset))
    budget = min(budget, len(candidates))
    if budget == len(candidates):
        return candidates

    rng = random.Random(tune_seed)
    return [candidates[0]] + rng.sample(candidates[1:], k=budget - 1)
