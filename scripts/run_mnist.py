#!/usr/bin/env python3
"""Convenience entry point for the MNIST configuration."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.argv[1:1] = ["--config", str(root / "configs" / "mnist.yaml")]
    runpy.run_module("scripts.run_experiment", run_name="__main__")
