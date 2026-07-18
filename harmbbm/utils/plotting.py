"""Publication-oriented plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _aligned_matrix(
    results: Sequence[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    if not results:
        raise ValueError("At least one result is required.")
    values = [
        np.asarray([row[metric] for row in result["history"]], dtype=np.float64)
        for result in results
    ]
    min_length = min(len(item) for item in values)
    epochs = np.asarray(
        [row["epoch"] for row in results[0]["history"][:min_length]],
        dtype=np.int64,
    )
    return epochs, np.vstack([item[:min_length] for item in values])


def plot_seed_curves(
    results: Sequence[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    log_scale: bool = False,
) -> None:
    epochs, matrix = _aligned_matrix(results, metric)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1 if matrix.shape[0] > 1 else 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for result, curve in zip(results, matrix):
        ax.plot(
            epochs,
            curve,
            linewidth=0.9,
            alpha=0.35,
            label=f"Seed {result['seed']}",
        )
    mean_line = ax.plot(epochs, mean, linewidth=2.2, label="Mean")[0]
    lower = mean - std
    if log_scale:
        lower = np.maximum(lower, np.finfo(float).tiny)
        ax.set_yscale("log")
    ax.fill_between(
        epochs,
        lower,
        mean + std,
        alpha=0.18,
        color=mean_line.get_color(),
        label="Mean ± SD",
    )
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


def plot_test_accuracy(
    results: Sequence[dict[str, Any]],
    title: str,
    output_stem: Path,
) -> None:
    values = [float(result["test_acc_at_best_val"]) for result in results]
    labels = [f"Seed {result['seed']}" for result in results]
    array = np.asarray(values, dtype=np.float64)
    mean_value = float(array.mean())
    std_value = float(array.std(ddof=1 if array.size > 1 else 0))
    plotted_values = values + [mean_value]
    plotted_labels = labels + ["Mean"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(plotted_labels, plotted_values)
    ax.errorbar(
        len(plotted_labels) - 1,
        mean_value,
        yerr=std_value,
        fmt="none",
        capsize=5,
        linewidth=1.2,
    )
    for bar, value in zip(bars, plotted_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    lower = max(0.0, min(plotted_values) - 2.0)
    upper = min(100.0, max(plotted_values) + 2.0)
    if upper <= lower:
        lower, upper = 0.0, 100.0
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
