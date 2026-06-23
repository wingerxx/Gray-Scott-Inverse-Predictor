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


class TargetScaler:
    """Per-parameter min-max scaler for regression targets.

    Scales each output dimension independently to ``[0, 1]`` based on the
    range of values seen during ``fit``. This ensures parameters with very
    different ranges (e.g. ``f`` spanning 0.05 and ``k`` spanning 0.004)
    contribute equally to the training loss.
    """

    def __init__(self):
        self.min_: np.ndarray = None
        self.scale_: np.ndarray = None

    def fit(self, targets: np.ndarray, eps: float = 1e-8) -> "TargetScaler":
        """Compute per-column min and range from ``targets`` (shape ``(N, D)``).

        Args:
            targets: Array of shape ``(N, D)`` where D is the number of targets.
            eps: Added to range to avoid division by zero for constant columns.
        """
        self.min_ = targets.min(axis=0).astype(np.float32)
        self.scale_ = (targets.max(axis=0) - self.min_ + eps).astype(np.float32)
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Scale targets to ``[0, 1]``."""
        return (targets - self.min_) / self.scale_

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        """Reverse the scaling back to original units."""
        return targets * self.scale_ + self.min_

    def save(self, path: str) -> None:
        """Save scaler state to a ``.npz`` file."""
        np.savez(path, min_=self.min_, scale_=self.scale_)

    @classmethod
    def load(cls, path: str) -> "TargetScaler":
        """Load scaler state from a ``.npz`` file."""
        data = np.load(path)
        scaler = cls()
        scaler.min_ = data["min_"]
        scaler.scale_ = data["scale_"]
        return scaler


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
