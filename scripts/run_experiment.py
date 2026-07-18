#!/usr/bin/env python3
"""Run the MNIST or PathMNIST optimizer comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from harmbbm.data import build_mnist_bundle, build_pathmnist_bundle
from harmbbm.engine import run_dataset_experiment
from harmbbm.optimizers import SUPPORTED_OPTIMIZERS
from harmbbm.utils import save_json


def _parse_csv_ints(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc


def _parse_csv_names(text: str) -> list[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one optimizer is required.")
    invalid = sorted(set(values) - set(SUPPORTED_OPTIMIZERS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported optimizers: {invalid}. Supported: {SUPPORTED_OPTIMIZERS}"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run HarmBBM, BBbound, SGDM, Adam, and AdaBelief on MNIST or "
            "PathMNIST."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--optimizers", type=_parse_csv_names)
    parser.add_argument("--seeds", type=_parse_csv_ints)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--do-search", action="store_true")
    parser.add_argument("--search-epochs", type=int)
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--tune-seed", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    experiment = dict(config["experiment"])
    search = dict(config.get("search", {}))
    optimizer_configs = {
        str(name).lower(): dict(values)
        for name, values in config["optimizers"].items()
    }

    if args.optimizers is not None:
        experiment["optimizers"] = args.optimizers
    if args.seeds is not None:
        experiment["seeds"] = args.seeds
    if args.epochs is not None:
        experiment["epochs"] = args.epochs
    if args.batch_size is not None:
        experiment["batch_size"] = args.batch_size
    if args.num_workers is not None:
        experiment["num_workers"] = args.num_workers
    if args.data_dir is not None:
        experiment["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        experiment["output_dir"] = str(args.output_dir)
    if args.no_download:
        experiment["download"] = False
    if args.no_augmentation:
        experiment["augmentation"] = False
    if args.do_search:
        search["enabled"] = True
    if args.search_epochs is not None:
        search["epochs"] = args.search_epochs
    if args.search_budget is not None:
        search["budget"] = args.search_budget
    if args.tune_seed is not None:
        search["tune_seed"] = args.tune_seed

    if args.quick:
        experiment["epochs"] = 2
        experiment["seeds"] = [42]
        search["epochs"] = 1
        search["budget"] = 2

    dataset = str(experiment["dataset"]).lower()
    optimizers = [str(name).lower() for name in experiment["optimizers"]]
    invalid = sorted(set(optimizers) - set(SUPPORTED_OPTIMIZERS))
    if invalid:
        raise ValueError(f"Unsupported optimizers: {invalid}")
    missing = [name for name in optimizers if name not in optimizer_configs]
    if missing:
        raise ValueError(f"Missing optimizer configurations: {missing}")

    data_dir = Path(experiment["data_dir"]).expanduser().resolve()
    output_dir = Path(experiment["output_dir"]).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        bundle = build_mnist_bundle(
            data_dir=data_dir,
            val_size=int(experiment.get("val_size", 10_000)),
            split_seed=int(experiment.get("split_seed", 42)),
            download=bool(experiment.get("download", True)),
        )
    elif dataset == "pathmnist":
        bundle = build_pathmnist_bundle(
            data_dir=data_dir,
            download=bool(experiment.get("download", True)),
            use_augmentation=bool(experiment.get("augmentation", True)),
        )
    else:
        raise ValueError("dataset must be 'mnist' or 'pathmnist'.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds: {experiment['seeds']}")
    print(f"Hyperparameter search: {bool(search.get('enabled', False))}")

    resolved_config = {
        "paper_title": (
            "HarmBBM: Harmonic Barzilai-Borwein Momentum for Deep Neural "
            "Network Training"
        ),
        "authors": ["Ali Raza", "Xinwei Liu", "Dileep Kumar"],
        "experiment": experiment,
        "search": search,
        "optimizers": {name: optimizer_configs[name] for name in optimizers},
        "device": str(device),
    }
    save_json(output_dir / dataset / "resolved_experiment_config.json", resolved_config)

    summaries = run_dataset_experiment(
        dataset=dataset,
        bundle=bundle,
        optimizer_names=optimizers,
        optimizer_configs=optimizer_configs,
        seeds=[int(seed) for seed in experiment["seeds"]],
        epochs=int(experiment["epochs"]),
        batch_size=int(experiment["batch_size"]),
        num_workers=int(experiment["num_workers"]),
        dropout=float(experiment.get("dropout", 0.0)),
        label_smoothing=float(experiment.get("label_smoothing", 0.0)),
        decay_epochs=[int(value) for value in experiment.get("decay_epochs", [])],
        decay_factor=float(experiment.get("decay_factor", 0.1)),
        deterministic=bool(experiment.get("deterministic", True)),
        device=device,
        output_root=output_dir,
        search_enabled=bool(search.get("enabled", False)),
        tune_seed=int(search.get("tune_seed", 42)),
        search_epochs=int(search.get("epochs", 30)),
        search_budget=int(search.get("budget", 50)),
    )
    print("\nCompleted summaries:")
    for summary in summaries:
        print(
            f"{summary['optimizer']}: "
            f"{summary['mean_test_acc_at_best_val']:.3f} ± "
            f"{summary['std_test_acc_at_best_val']:.3f}%"
        )


if __name__ == "__main__":
    main()
