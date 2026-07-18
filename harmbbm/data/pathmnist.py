"""PathMNIST dataset construction."""

from __future__ import annotations

from pathlib import Path

from torchvision import transforms

from .common import DatasetBundle


def build_pathmnist_bundle(
    data_dir: Path,
    download: bool = True,
    use_augmentation: bool = True,
) -> DatasetBundle:
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise RuntimeError(
            "PathMNIST requires medmnist. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    info = INFO["pathmnist"]
    data_class = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                normalization,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(), normalization]
        )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalization])

    common = {
        "root": str(data_dir),
        "download": download,
        "as_rgb": True,
    }
    train_set = data_class(split="train", transform=train_transform, **common)
    val_set = data_class(split="val", transform=eval_transform, **common)
    test_set = data_class(split="test", transform=eval_transform, **common)

    return DatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        num_classes=num_classes,
        in_channels=3,
        model_name="ResNet18",
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
    )
