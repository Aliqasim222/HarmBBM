#!/usr/bin/env python3
"""Create cross-optimizer comparison figures from saved result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METHODS = ["sgdm", "adam", "adabelief", "bbbound", "harmbbm"]
DISPLAY_NAMES = {
    "sgdm": "SGDM",
    "adam": "Adam",
    "adabelief": "AdaBelief",
    "bbbound": "BBbound",
    "harmbbm": "HarmBBM",
}


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"No results found in {path}")
    return payload


def _best_seed_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda item: (
            float(item["best_val_acc"]),
            -float(item["best_val_loss"]),
            -int(item["best_epoch_by_validation"]),
        ),
    )


def _plot_curves(
    selected: dict[str, dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    log_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, result in selected.items():
        history = result["history"]
        epochs = [int(row["epoch"]) for row in history]
        values = [float(row[metric]) for row in history]
        ax.plot(epochs, values, linewidth=1.8, label=DISPLAY_NAMES[method])
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_test_bar(
    all_results: dict[str, list[dict[str, Any]]],
    title: str,
    output_stem: Path,
) -> None:
    methods = list(all_results)
    means = []
    stds = []
    for method in methods:
        values = np.asarray(
            [float(item["test_acc_at_best_val"]) for item in all_results[method]],
            dtype=np.float64,
        )
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1 if values.size > 1 else 0)))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [DISPLAY_NAMES[method] for method in methods]
    bars = ax.bar(labels, means, yerr=stds, capsize=5)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean,
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    lower = max(0.0, min(means) - 3.0)
    upper = min(100.0, max(means) + 3.0)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("./results"))
    parser.add_argument("--dataset", choices=["mnist", "pathmnist"], required=True)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method directory names.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./figures"))
    args = parser.parse_args()

    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    all_results: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method not in DISPLAY_NAMES:
            raise ValueError(f"Unknown method: {method}")
        path = args.results_root / args.dataset / method / "all_seed_results.json"
        all_results[method] = _load_results(path)
        selected[method] = _best_seed_result(all_results[method])

    dataset_label = "MNIST / MLP-1024x1024" if args.dataset == "mnist" else "PathMNIST / ResNet18"
    _plot_curves(
        selected,
        metric="train_loss",
        ylabel="Training loss",
        title=f"{dataset_label}: Training Loss",
        output_stem=args.out_dir / f"{args.dataset}_training_loss_best_seed_log",
        log_scale=True,
    )
    _plot_curves(
        selected,
        metric="val_acc",
        ylabel="Validation accuracy (%)",
        title=f"{dataset_label}: Validation Accuracy",
        output_stem=args.out_dir / f"{args.dataset}_validation_accuracy_best_seed",
        log_scale=False,
    )
    _plot_test_bar(
        all_results,
        title=f"{dataset_label}: Three-Seed Test Accuracy",
        output_stem=args.out_dir / f"{args.dataset}_test_accuracy_mean_std",
    )
    print(f"Figures saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
