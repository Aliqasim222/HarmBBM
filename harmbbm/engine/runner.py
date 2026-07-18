"""Dataset-level experiment runner with optional validation search."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from harmbbm.data import DatasetBundle
from harmbbm.search import build_search_candidates
from harmbbm.utils import save_json, save_pickle, save_rows_csv

from .core import run_single
from .results import save_method_results


def _trial_key(trial: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(trial["best_val_acc"]),
        -float(trial["best_val_loss"]),
        -int(trial["best_epoch_by_validation"]),
    )


def _save_search_outputs(
    trials: list[dict[str, Any]],
    selected: dict[str, Any],
    output_dir: Path,
) -> None:
    compact_rows: list[dict[str, Any]] = []
    for trial in trials:
        row = {
            "config_index": trial["config_index"],
            "dataset": trial["dataset"],
            "optimizer": trial["optimizer"],
            "tune_seed": trial["tune_seed"],
            "search_epochs": trial["search_epochs"],
            "best_val_acc": trial["best_val_acc"],
            "best_val_loss": trial["best_val_loss"],
            "best_epoch_by_validation": trial["best_epoch_by_validation"],
            "elapsed_sec": trial["elapsed_sec"],
        }
        row.update({f"config_{key}": value for key, value in trial["config"].items()})
        compact_rows.append(row)

    save_rows_csv(output_dir / "search_trials.csv", compact_rows)
    save_json(output_dir / "search_trials.json", trials)
    save_pickle(output_dir / "search_trials.pkl", trials)

    selected_record = {
        "selection_rule": (
            "Highest validation accuracy; ties are broken by lower validation "
            "loss and then by the earlier best-validation epoch."
        ),
        "test_set_used_for_selection": False,
        "search_budget_completed": len(trials),
        "selected_config_index": selected["config_index"],
        "selected_config": selected["config"],
        "best_val_acc": selected["best_val_acc"],
        "best_val_loss": selected["best_val_loss"],
        "best_epoch_by_validation": selected["best_epoch_by_validation"],
        "tune_seed": selected["tune_seed"],
        "search_epochs": selected["search_epochs"],
    }
    selected_csv = {
        key: value
        for key, value in selected_record.items()
        if key != "selected_config"
    }
    selected_csv.update(
        {
            f"selected_{key}": value
            for key, value in selected_record["selected_config"].items()
        }
    )
    save_rows_csv(output_dir / "selected_hyperparameters.csv", [selected_csv])
    save_json(output_dir / "selected_hyperparameters.json", selected_record)
    save_pickle(output_dir / "selected_hyperparameters.pkl", selected_record)


def run_hyperparameter_search(
    *,
    dataset: str,
    optimizer_name: str,
    default_config: dict[str, Any],
    bundle: DatasetBundle,
    tune_seed: int,
    search_epochs: int,
    search_budget: int,
    batch_size: int,
    num_workers: int,
    dropout: float,
    label_smoothing: float,
    decay_epochs: list[int],
    decay_factor: float,
    deterministic: bool,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = build_search_candidates(
        optimizer_name=optimizer_name,
        dataset=dataset,
        default_config=default_config,
        budget=search_budget,
        tune_seed=tune_seed,
    )
    print(
        f"\nSearch: {dataset}/{optimizer_name}, {len(candidates)} configurations, "
        f"{search_epochs} epochs each, seed {tune_seed}."
    )

    trials: list[dict[str, Any]] = []
    for config_index, config in enumerate(candidates, start=1):
        print(
            f"\n[search {config_index:03d}/{len(candidates):03d}] "
            f"{json.dumps(config, sort_keys=True)}"
        )
        result = run_single(
            dataset=dataset,
            optimizer_name=optimizer_name,
            optimizer_config=config,
            bundle=bundle,
            seed=tune_seed,
            epochs=search_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout=dropout,
            label_smoothing=label_smoothing,
            decay_epochs=decay_epochs,
            decay_factor=decay_factor,
            deterministic=deterministic,
            device=device,
            run_dir=output_dir / f"trial_{config_index:03d}",
            evaluate_test=False,
            verbose=False,
        )
        trial = {
            "config_index": config_index,
            "dataset": dataset,
            "optimizer": optimizer_name,
            "config": copy.deepcopy(config),
            "tune_seed": tune_seed,
            "search_epochs": search_epochs,
            "best_val_acc": result["best_val_acc"],
            "best_val_loss": result["best_val_loss"],
            "best_epoch_by_validation": result["best_epoch_by_validation"],
            "elapsed_sec": result["elapsed_sec"],
            "history": result["history"],
        }
        trials.append(trial)
        selected = max(trials, key=_trial_key)
        _save_search_outputs(trials, selected, output_dir)
        print(
            f"[search {config_index:03d}] best val acc "
            f"{trial['best_val_acc']:.3f}% at epoch "
            f"{trial['best_epoch_by_validation']}"
        )

    selected = max(trials, key=_trial_key)
    _save_search_outputs(trials, selected, output_dir)
    print("Selected configuration:")
    print(json.dumps(selected["config"], indent=2, sort_keys=True))
    return dict(selected["config"])


def run_dataset_experiment(
    *,
    dataset: str,
    bundle: DatasetBundle,
    optimizer_names: list[str],
    optimizer_configs: dict[str, dict[str, Any]],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    num_workers: int,
    dropout: float,
    label_smoothing: float,
    decay_epochs: list[int],
    decay_factor: float,
    deterministic: bool,
    device: torch.device,
    output_root: Path,
    search_enabled: bool,
    tune_seed: int,
    search_epochs: int,
    search_budget: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for optimizer_name in optimizer_names:
        method_dir = output_root / dataset / optimizer_name
        method_dir.mkdir(parents=True, exist_ok=True)
        selected_config = dict(optimizer_configs[optimizer_name])

        if search_enabled:
            selected_config = run_hyperparameter_search(
                dataset=dataset,
                optimizer_name=optimizer_name,
                default_config=selected_config,
                bundle=bundle,
                tune_seed=tune_seed,
                search_epochs=search_epochs,
                search_budget=search_budget,
                batch_size=batch_size,
                num_workers=num_workers,
                dropout=dropout,
                label_smoothing=label_smoothing,
                decay_epochs=decay_epochs,
                decay_factor=decay_factor,
                deterministic=deterministic,
                device=device,
                output_dir=method_dir / "search",
            )

        run_config = {
            "paper_title": (
                "HarmBBM: Harmonic Barzilai-Borwein Momentum for Deep Neural "
                "Network Training"
            ),
            "dataset": dataset,
            "optimizer": optimizer_name,
            "model": bundle.model_name,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "selected_config": selected_config,
            "hyperparameter_search": {
                "enabled": search_enabled,
                "tune_seed": tune_seed,
                "search_epochs": search_epochs,
                "search_budget": search_budget,
                "test_set_used_for_selection": False,
            },
            "checkpoint_selection": (
                "Highest validation accuracy; ties broken by lower validation "
                "loss. Test data are evaluated once after restoring the selected "
                "checkpoint."
            ),
        }
        save_json(method_dir / "run_config.json", run_config)
        save_pickle(method_dir / "run_config.pkl", run_config)

        results: list[dict[str, Any]] = []
        for seed in seeds:
            result = run_single(
                dataset=dataset,
                optimizer_name=optimizer_name,
                optimizer_config=selected_config,
                bundle=bundle,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                dropout=dropout,
                label_smoothing=label_smoothing,
                decay_epochs=decay_epochs,
                decay_factor=decay_factor,
                deterministic=deterministic,
                device=device,
                run_dir=method_dir / f"seed_{seed}",
                evaluate_test=True,
                verbose=True,
            )
            results.append(result)

        summary = save_method_results(results, method_dir)
        summary["search_enabled"] = search_enabled
        summaries.append(summary)

    save_rows_csv(output_root / dataset / "optimizer_summary.csv", summaries)
    save_json(output_root / dataset / "optimizer_summary.json", summaries)
    save_pickle(output_root / dataset / "optimizer_summary.pkl", summaries)
    return summaries
