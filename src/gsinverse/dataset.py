"""Torch ``Dataset`` implementations over the cached Gray-Scott data.

Supports an 80/20 train/val split performed on the ``(f, k)`` pairs
themselves (not on individual generated samples), so the same parameter
pair never appears in both splits.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .datagen import generate_and_cache
from .utils import TargetScaler


class CachedGrayScottDataset(Dataset):
    """Dataset backed by a pre-generated ``.npz`` cache."""

    def __init__(self, images: np.ndarray, targets: np.ndarray, scaler: Optional[TargetScaler] = None):
        self.images = images
        # Store raw targets; apply scaler at access time so the scaler can be
        # swapped or inspected without rebuilding the dataset.
        self.targets_raw = targets
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])
        target = self.targets_raw[idx]
        if self.scaler is not None:
            target = self.scaler.transform(target[None])[0]
        return image, torch.from_numpy(target.astype(np.float32))



def split_pairs(
    fk_pairs: List[Tuple[float, float]], config: dict
) -> Tuple[List[int], List[int]]:
    """Split ``(f, k)`` pair indices into train/val sets.

    Args:
        fk_pairs: List of ``(f, k)`` pairs.
        config: Full configuration dict (uses ``data.val_fraction`` and
            ``data.split_seed``).

    Returns:
        A tuple ``(train_idx, val_idx)`` of pair indices.
    """
    indices = list(range(len(fk_pairs)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config["data"]["val_fraction"],
        random_state=config["data"]["split_seed"],
    )
    return train_idx, val_idx


def build_train_val_datasets(
    fk_pairs: List[Tuple[float, float]],
    config: dict,
    scaler: Optional[TargetScaler] = None,
    show_progress: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Build the train and validation datasets, split by ``(f, k)`` pair.

    Args:
        fk_pairs: List of ``(f, k)`` pairs (e.g. the converged pairs produced
            by ``scripts/generate_grid_pairs.py``).
        config: Full configuration dict.
        scaler: Optional :class:`~gsinverse.utils.TargetScaler`. When provided,
            targets are scaled before being returned by ``__getitem__``. Fit
            the scaler on training targets before passing it here.
        show_progress: If ``True``, show a progress bar during cache generation.

    Returns:
        A tuple ``(train_dataset, val_dataset)``.
    """
    train_idx, val_idx = split_pairs(fk_pairs, config)

    cache_path = generate_and_cache(fk_pairs, config, show_progress=show_progress)
    cache = np.load(cache_path)

    images = cache["images"]
    targets = cache["targets"]
    pair_idx = cache["pair_idx"]

    train_mask = np.isin(pair_idx, train_idx)
    val_mask = ~train_mask

    train_ds = CachedGrayScottDataset(images[train_mask], targets[train_mask], scaler=scaler)
    val_ds = CachedGrayScottDataset(images[val_mask], targets[val_mask], scaler=scaler)
    return train_ds, val_ds
