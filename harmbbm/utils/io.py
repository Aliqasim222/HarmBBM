"""Serialization helpers used by the experiment scripts."""

from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(obj), file, indent=2, ensure_ascii=False)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot save an empty CSV: {path}")

    fieldnames: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in observed:
                fieldnames.append(key)
                observed.add(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: (
                    json.dumps(json_ready(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else json_ready(value)
                )
                for key, value in row.items()
            }
            writer.writerow(flattened)
