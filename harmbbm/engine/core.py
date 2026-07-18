"""Training, validation, and single-run execution."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from harmbbm.data import DatasetBundle, make_loaders
from harmbbm.models import MLP1024, build_resnet18
from harmbbm.optimizers import BBbound, HarmBBM, build_optimizer
from harmbbm.utils import save_json, save_pickle, save_rows_csv, set_global_seed


def build_model(dataset: str, bundle: DatasetBundle, dropout: float) -> nn.Module:
    if dataset == "mnist":
        return MLP1024(num_classes=bundle.num_classes, dropout=dropout)
    if dataset == "pathmnist":
        return build_resnet18(
            num_classes=bundle.num_classes,
            in_channels=bundle.in_channels,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def scheduled_base_lr(
    initial_lr: float,
    epoch: int,
    milestones: Sequence[int],
    decay_factor: float,
) -> float:
    """Return the base rate for a one-indexed epoch."""
    reductions = sum(1 for milestone in milestones if milestone < epoch)
    return float(initial_lr) * float(decay_factor) ** reductions


def _prepare_targets(targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    return targets.view(-1).long().to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    if isinstance(optimizer, BBbound):
        optimizer.begin_epoch()

    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = _prepare_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("Training loader produced no samples.")
    return total_loss / total, 100.0 * total_correct / total


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = _prepare_targets(targets, device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    return total_loss / total, 100.0 * total_correct / total


def _safe_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_single(
    *,
    dataset: str,
    optimizer_name: str,
    optimizer_config: dict[str, Any],
    bundle: DatasetBundle,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    dropout: float,
    label_smoothing: float,
    decay_epochs: Sequence[int],
    decay_factor: float,
    deterministic: bool,
    device: torch.device,
    run_dir: Path,
    evaluate_test: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    set_global_seed(seed, deterministic=deterministic)
    train_loader, val_loader, test_loader = make_loaders(
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    model = build_model(dataset, bundle, dropout=dropout).to(device)
    optimizer = build_optimizer(
        optimizer_name,
        model,
        optimizer_config,
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler = None
    if optimizer_name in {"sgdm", "adam", "adabelief"} and decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(decay_epochs),
            gamma=decay_factor,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_validation_model.pt"
    history: list[dict[str, Any]] = []
    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        used_lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        diagnostics: dict[str, Any] = {}
        if isinstance(optimizer, HarmBBM):
            next_base = scheduled_base_lr(
                float(optimizer_config["lr"]),
                epoch + 1,
                decay_epochs,
                decay_factor,
            )
            diagnostics = optimizer.end_epoch(next_base)
        elif isinstance(optimizer, BBbound):
            milestone_decay = decay_factor if epoch in decay_epochs else 1.0
            diagnostics = optimizer.end_epoch(
                epoch_index=epoch,
                steps_per_epoch=len(train_loader),
                decay_factor=milestone_decay,
            )
        elif scheduler is not None:
            scheduler.step()
            diagnostics = {
                "used_lr": used_lr,
                "next_lr": float(optimizer.param_groups[0]["lr"]),
            }
        else:
            diagnostics = {"used_lr": used_lr, "next_lr": used_lr}

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **diagnostics,
            "epoch_elapsed_sec": time.perf_counter() - epoch_start,
            "total_elapsed_sec": time.perf_counter() - start_time,
        }
        history.append(row)

        improved = val_acc > best_val_acc or (
            math.isclose(val_acc, best_val_acc, rel_tol=0.0, abs_tol=1e-12)
            and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            if evaluate_test:
                torch.save(
                    {
                        "paper_method": "HarmBBM" if optimizer_name == "harmbbm" else optimizer_name,
                        "dataset": dataset,
                        "model": bundle.model_name,
                        "optimizer": optimizer_name,
                        "optimizer_config": copy.deepcopy(optimizer_config),
                        "seed": seed,
                        "epoch": epoch,
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )

        if verbose:
            print(
                f"[{dataset}/{optimizer_name}/seed={seed}] "
                f"epoch {epoch:03d}/{epochs} | "
                f"train {train_loss:.5f}, {train_acc:.3f}% | "
                f"val {val_loss:.5f}, {val_acc:.3f}% | "
                f"lr {used_lr:.6g} -> {float(optimizer.param_groups[0]['lr']):.6g}"
            )

    test_loss = None
    test_acc = None
    if evaluate_test:
        if best_epoch < 0 or not checkpoint_path.exists():
            raise RuntimeError("No validation checkpoint was saved.")
        checkpoint = _safe_load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    result = {
        "dataset": dataset,
        "optimizer": optimizer_name,
        "method_name": "HarmBBM" if optimizer_name == "harmbbm" else optimizer_name,
        "model": bundle.model_name,
        "seed": int(seed),
        "epochs": int(epochs),
        "config": copy.deepcopy(optimizer_config),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch_by_validation": int(best_epoch),
        "test_acc_at_best_val": None if test_acc is None else float(test_acc),
        "test_loss_at_best_val": None if test_loss is None else float(test_loss),
        "elapsed_sec": float(time.perf_counter() - start_time),
        "checkpoint": str(checkpoint_path) if evaluate_test else None,
        "history": history,
    }

    if evaluate_test:
        save_rows_csv(run_dir / "epoch_history.csv", history)
        save_json(run_dir / "epoch_history.json", history)
        save_pickle(run_dir / "epoch_history.pkl", history)
        save_json(run_dir / "result.json", result)
        save_pickle(run_dir / "result.pkl", result)

    del optimizer, model, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
