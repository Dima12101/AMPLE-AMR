"""Utility helpers used across the simulation package."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory when it does not exist and return it as a path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to a closed interval."""

    return max(lower, min(upper, value))


def percentile(values: list[float], q: float) -> float:
    """Compute a percentile for a non-empty list."""

    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_div(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning zero on division by zero."""

    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator
