"""Build and cache the (image, [f, k]) training dataset.

For each ``(f, k)`` pair drawn from the source HDF5 dataset, this module
runs the forward simulator (with fresh random seeds) to produce one or more
``(3, 64, 64)`` model-input tensors, and caches the resulting arrays to disk
so training does not need to re-simulate every epoch.
"""

import hashlib
import json
import os
import random
from typing import List, Tuple

import numpy as np

from .simulator import simulate_gray_scott
from .utils import normalize_minmax


def cache_key(config: dict, fk_pairs: List[Tuple[float, float]]) -> str:
    """Compute a deterministic hash identifying a dataset configuration.

    The hash covers everything that affects the generated samples: the
    simulator settings, the number of seeds per parameter pair, and the
    exact set of ``(f, k)`` pairs.

    Args:
        config: The full configuration dict (only relevant sub-keys are used).
        fk_pairs: The list of ``(f, k)`` pairs the dataset is built from.

    Returns:
        A short hex digest suitable for use as a cache filename.
    """
    sim_cfg = config["simulator"]
    payload = {
        "simulator": {
            "du": sim_cfg["du"],
            "dv": sim_cfg["dv"],
            "size": sim_cfg["size"],
            "iterations": sim_cfg["iterations"],
            "patch_radius": sim_cfg["patch_radius"],
            "patch_prob": sim_cfg["patch_prob"],
        },
        "seeds_per_param": config["data"]["seeds_per_param"],
        "fk_pairs": [[float(f), float(k)] for f, k in fk_pairs],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def generate_sample(
    f: float, k: float, config: dict, seed: int
) -> np.ndarray:
    """Run one simulation and build a normalized ``(3, 64, 64)`` input tensor.

    The three channels are: final U field, final V field, and the initial
    V field (encoding the seed/initial-condition). Each channel is
    independently min-max normalized to ``[-1, 1]``.

    Args:
        f: Feed rate.
        k: Kill rate.
        config: Full configuration dict (uses the ``simulator`` section).
        seed: Random seed for the initial conditions.

    Returns:
        A ``(3, 64, 64)`` float32 array.
    """
    sim_cfg = config["simulator"]

    u_final, v_final, _, v_initial = simulate_gray_scott(
        du=sim_cfg["du"],
        dv=sim_cfg["dv"],
        f=f,
        k=k,
        iterations=sim_cfg["iterations"],
        size=sim_cfg["size"],
        patch_radius=sim_cfg["patch_radius"],
        patch_prob=sim_cfg["patch_prob"],
        seed=seed,
        device="cpu",
        return_initial=True,
    )

    channels = [
        normalize_minmax(u_final.numpy()),
        normalize_minmax(v_final.numpy()),
        normalize_minmax(v_initial.numpy()),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def build_dataset(
    fk_pairs: List[Tuple[float, float]], config: dict, show_progress: bool = True
) -> dict:
    """Generate samples for every ``(f, k)`` pair with multiple random seeds.

    Args:
        fk_pairs: List of ``(f, k)`` pairs.
        config: Full configuration dict.
        show_progress: If ``True``, display a tqdm progress bar.

    Returns:
        A dict with keys:
            - ``images``: ``(N, 3, 64, 64)`` float32 array.
            - ``targets``: ``(N, 2)`` float32 array of ``[f, k]``.
            - ``pair_idx``: ``(N,)`` int64 array indexing into ``fk_pairs``,
              used to split by parameter pair without leakage.
    """
    seeds_per_param = config["data"]["seeds_per_param"]

    images = []
    targets = []
    pair_idx = []

    iterator = range(len(fk_pairs))
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Generating dataset")

    for i in iterator:
        f, k = fk_pairs[i]
        for _ in range(seeds_per_param):
            seed = random.randint(0, 2**31 - 1)
            images.append(generate_sample(f, k, config, seed))
            targets.append([f, k])
            pair_idx.append(i)

    return {
        "images": np.stack(images, axis=0),
        "targets": np.array(targets, dtype=np.float32),
        "pair_idx": np.array(pair_idx, dtype=np.int64),
    }


def generate_and_cache(
    fk_pairs: List[Tuple[float, float]], config: dict, show_progress: bool = True
) -> str:
    """Build the dataset (if not already cached) and return the cache path.

    Args:
        fk_pairs: List of ``(f, k)`` pairs.
        config: Full configuration dict.
        show_progress: If ``True``, display a tqdm progress bar during generation.

    Returns:
        Path to the ``.npz`` cache file.
    """
    cache_dir = config["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    key = cache_key(config, fk_pairs)
    cache_path = os.path.join(cache_dir, f"dataset_{key}.npz")

    if os.path.exists(cache_path):
        return cache_path

    data = build_dataset(fk_pairs, config, show_progress=show_progress)
    fk_array = np.array(fk_pairs, dtype=np.float32)

    np.savez_compressed(
        cache_path,
        images=data["images"],
        targets=data["targets"],
        pair_idx=data["pair_idx"],
        fk_pairs=fk_array,
    )
    return cache_path
