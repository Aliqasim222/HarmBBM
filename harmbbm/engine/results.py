"""Aggregation and output generation for multi-seed experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from harmbbm.utils import save_json, save_pickle, save_rows_csv
from harmbbm.utils.plotting import plot_seed_curves, plot_test_accuracy


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    ddof = 1 if array.size > 1 else 0
    return float(array.mean()), float(array.std(ddof=ddof))


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one result is required.")

    best_val_mean, best_val_std = mean_std(
        [float(result["best_val_acc"]) for result in results]
    )
    test_acc_mean, test_acc_std = mean_std(
        [float(result["test_acc_at_best_val"]) for result in results]
    )
    test_loss_mean, test_loss_std = mean_std(
        [float(result["test_loss_at_best_val"]) for result in results]
    )
    best_epoch_mean, best_epoch_std = mean_std(
        [float(result["best_epoch_by_validation"]) for result in results]
    )
    elapsed_mean, elapsed_std = mean_std(
        [float(result["elapsed_sec"]) for result in results]
    )

    first = results[0]
    return {
        "dataset": first["dataset"],
        "model": first["model"],
        "optimizer": first["optimizer"],
        "method_name": first["method_name"],
        "n_seeds": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "mean_best_val_acc": best_val_mean,
        "std_best_val_acc": best_val_std,
        "mean_test_acc_at_best_val": test_acc_mean,
        "std_test_acc_at_best_val": test_acc_std,
        "mean_test_loss_at_best_val": test_loss_mean,
        "std_test_loss_at_best_val": test_loss_std,
        "mean_best_epoch": best_epoch_mean,
        "std_best_epoch": best_epoch_std,
        "mean_elapsed_sec": elapsed_mean,
        "std_elapsed_sec": elapsed_std,
        "selected_config": first["config"],
    }


def save_method_results(
    results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    summary = aggregate_results(results)
    method = str(summary["optimizer"])
    dataset = str(summary["dataset"])

    compact_rows = [
        {
            "dataset": result["dataset"],
            "optimizer": result["optimizer"],
            "model": result["model"],
            "seed": result["seed"],
            "epochs": result["epochs"],
            "best_val_acc": result["best_val_acc"],
            "best_val_loss": result["best_val_loss"],
            "best_epoch_by_validation": result["best_epoch_by_validation"],
            "test_acc_at_best_val": result["test_acc_at_best_val"],
            "test_loss_at_best_val": result["test_loss_at_best_val"],
            "elapsed_sec": result["elapsed_sec"],
        }
        for result in results
    ]

    save_rows_csv(output_dir / "all_seed_results.csv", compact_rows)
    save_json(output_dir / "all_seed_results.json", list(results))
    save_pickle(output_dir / "all_seed_results.pkl", list(results))
    save_rows_csv(output_dir / "mean_std_summary.csv", [summary])
    save_json(output_dir / "mean_std_summary.json", summary)
    save_pickle(output_dir / "mean_std_summary.pkl", summary)

    figures_dir = output_dir / "figures"
    plot_seed_curves(
        results,
        metric="train_loss",
        ylabel="Training loss",
        title=f"{dataset.upper()} / {summary['model']}: {summary['method_name']} training loss",
        output_stem=figures_dir / f"{dataset}_{method}_training_loss",
        log_scale=True,
    )
    plot_seed_curves(
        results,
        metric="val_acc",
        ylabel="Validation accuracy (%)",
        title=f"{dataset.upper()} / {summary['model']}: {summary['method_name']} validation accuracy",
        output_stem=figures_dir / f"{dataset}_{method}_validation_accuracy",
    )
    plot_test_accuracy(
        results,
        title=f"{dataset.upper()} / {summary['model']}: {summary['method_name']} test accuracy",
        output_stem=figures_dir / f"{dataset}_{method}_test_accuracy",
    )
    return summary
