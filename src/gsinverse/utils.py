"""Shared helpers: device selection, normalization, seeding, and config loading."""

import random
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def get_device() -> torch.device:
    """Select the best available torch device.

    Preference order: CUDA, then Apple Silicon MPS, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-array min-max normalization to ``[-1, 1]``.

    Args:
        x: Input array.
        eps: Small constant to avoid division by zero for constant arrays.

    Returns:
        Array with the same shape as ``x``, scaled to ``[-1, 1]``.
    """
    x_min = x.min()
    x_max = x.max()
    return 2 * (x - x_min) / (x_max - x_min + eps) - 1


def load_config(path: str, overrides: List[str] = None) -> Dict[str, Any]:
    """Load a YAML config file, applying optional ``key.path=value`` overrides.

    Args:
        path: Path to the YAML config file.
        overrides: A list of strings of the form ``"a.b.c=value"``, used to
            override nested config values from the CLI. Values are parsed as
            YAML scalars (so ``"true"``, ``"1.0e-4"``, ``"5"`` etc. work as
            expected).

    Returns:
        The (possibly overridden) config dict.
    """
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)

    for override in overrides or []:
        key_path, _, raw_value = override.partition("=")
        value = yaml.safe_load(raw_value)

        keys = key_path.split(".")
        node = config
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value

    return config
