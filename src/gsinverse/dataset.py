"""Torch ``Dataset`` implementations over the cached Gray-Scott data.

Provides a cached dataset (the default, fast path) and an on-the-fly
dataset that regenerates samples with fresh random seeds every time they
are accessed. Both support an 80/20 train/val split performed on the
``(f, k)`` pairs themselves (not on individual generated samples), so the
same parameter pair never appears in both splits.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .datagen import generate_and_cache, generate_sample
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


class OnTheFlyGrayScottDataset(Dataset):
    """Dataset that regenerates samples with fresh random seeds on access.

    Each call to ``__getitem__`` runs a new simulation for the
    ``(f, k)`` pair at ``idx``, using a freshly drawn random seed. This
    provides maximal seed augmentation at the cost of simulation time.
    """

    def __init__(self, fk_pairs: List[Tuple[float, float]], config: dict, scaler: Optional[TargetScaler] = None):
        self.fk_pairs = fk_pairs
        self.config = config
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.fk_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f, k = self.fk_pairs[idx]
        seed = random.randint(0, 2**31 - 1)
        image = generate_sample(f, k, self.config, seed)
        target = np.array([[f, k]], dtype=np.float32)
        if self.scaler is not None:
            target = self.scaler.transform(target)
        return torch.from_numpy(image), torch.from_numpy(target[0])


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
        fk_pairs: List of ``(f, k)`` pairs (e.g. from
            :func:`gsinverse.params.extract_fk_pairs`).
        config: Full configuration dict.
        scaler: Optional :class:`~gsinverse.utils.TargetScaler`. When provided,
            targets are scaled before being returned by ``__getitem__``. Fit
            the scaler on training targets before passing it here.
        show_progress: If ``True``, show a progress bar during cache generation
            (only used when ``data.on_the_fly`` is ``False``).

    Returns:
        A tuple ``(train_dataset, val_dataset)``.
    """
    train_idx, val_idx = split_pairs(fk_pairs, config)

    if config["data"]["on_the_fly"]:
        train_pairs = [fk_pairs[i] for i in train_idx]
        val_pairs = [fk_pairs[i] for i in val_idx]
        return (
            OnTheFlyGrayScottDataset(train_pairs, config, scaler=scaler),
            OnTheFlyGrayScottDataset(val_pairs, config, scaler=scaler),
        )

    cache_path = generate_and_cache(fk_pairs, config, show_progress=show_progress)
    cache = np.load(cache_path)

    images = cache["images"]
    targets = cache["targets"]
    pair_idx = cache["pair_idx"]

    train_pair_set = set(train_idx)
    val_pair_set = set(val_idx)

    train_mask = np.array([p in train_pair_set for p in pair_idx])
    val_mask = np.array([p in val_pair_set for p in pair_idx])

    train_ds = CachedGrayScottDataset(images[train_mask], targets[train_mask], scaler=scaler)
    val_ds = CachedGrayScottDataset(images[val_mask], targets[val_mask], scaler=scaler)
    return train_ds, val_ds
