"""Common dataset and DataLoader utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetBundle:
    train_set: Dataset
    val_set: Dataset
    test_set: Dataset
    num_classes: int
    in_channels: int
    model_name: str
    train_size: int
    val_size: int
    test_size: int


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        bundle.train_set,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        bundle.val_set,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        bundle.test_set,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader
