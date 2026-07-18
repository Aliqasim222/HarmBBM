"""MNIST dataset construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from .common import DatasetBundle


def build_mnist_bundle(
    data_dir: Path,
    val_size: int = 10_000,
    split_seed: int = 42,
    download: bool = True,
) -> DatasetBundle:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=download,
    )
    test_set = datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=download,
    )
    if not 0 < val_size < len(full_train):
        raise ValueError(f"val_size must be in [1, {len(full_train) - 1}].")

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(full_train), generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    return DatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        num_classes=10,
        in_channels=1,
        model_name="MLP-1024x1024",
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
    )
