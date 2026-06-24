"""Build and cache the (image, [f, k]) training dataset.

For each ``(f, k)`` pair, this module runs the forward simulator with
deterministic seeds to produce one or more ``(3, 64, 64)`` model-input
tensors, and caches the resulting arrays to disk so training does not need
to re-simulate every epoch.

Seeds are assigned deterministically as ``pair_index * seeds_per_param + s``
so the cache key fully determines reproducibility — no global RNG state needed.
"""

import hashlib
import json
import os
from typing import List, Tuple

import numpy as np

from .simulator import simulate_gray_scott
from .utils import normalize_minmax


def cache_key(
    config: dict, fk_pairs: List[Tuple[float, float]], source: str = "single_phase"
) -> str:
    """Compute a deterministic hash identifying a dataset configuration.

    The hash covers everything that affects the generated samples: the
    generation method (``source``), the simulator settings, the number of
    seeds per parameter pair, the grid convergence settings (when present),
    and the exact set of ``(f, k)`` pairs.

    Args:
        config: The full configuration dict (only relevant sub-keys are used).
        fk_pairs: The list of ``(f, k)`` pairs the dataset is built from.
        source: A short tag identifying which pipeline produced the images.
            ``"single_phase"`` for :func:`build_dataset` (one
            :func:`simulate_gray_scott` run per seed) and ``"grid"`` for the
            two-phase convergence/snapshot path in
            ``scripts/generate_grid_pairs.py``. The two paths produce
            different images, so the tag keeps their caches from colliding.

    Returns:
        A short hex digest suitable for use as a cache filename.
    """
    sim_cfg = config["simulator"]
    payload = {
        "source": source,
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

    # The grid/snapshot path's images depend on the convergence settings;
    # fold them in (when present) so changing them busts the cache.
    conv_cfg = config.get("grid", {}).get("convergence")
    if conv_cfg is not None:
        payload["convergence"] = {
            "iterations": conv_cfg.get("iterations"),
            "stability_steps": conv_cfg.get("stability_steps"),
            "n_seeds": conv_cfg.get("n_seeds"),
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

    n_pairs = len(fk_pairs)
    n_samples = n_pairs * seeds_per_param
    size = config["simulator"]["size"]

    images = np.empty((n_samples, 3, size, size), dtype=np.float32)
    targets = np.empty((n_samples, 2), dtype=np.float32)
    pair_idx = np.empty(n_samples, dtype=np.int64)

    iterator = range(n_pairs)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Generating dataset")

    for i in iterator:
        f, k = fk_pairs[i]
        for s in range(seeds_per_param):
            seed = i * seeds_per_param + s
            out = i * seeds_per_param + s
            images[out] = generate_sample(f, k, config, seed)
            targets[out] = [f, k]
            pair_idx[out] = i

    return {
        "images": images,
        "targets": targets,
        "pair_idx": pair_idx,
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

    # Prefer a cache pre-generated by the grid/snapshot path (the images the
    # convergence check produced); only fall back to a single-phase build if
    # no grid cache exists for this configuration.
    grid_key = cache_key(config, fk_pairs, source="grid")
    grid_path = os.path.join(cache_dir, f"dataset_{grid_key}.npz")
    if os.path.exists(grid_path):
        return grid_path

    key = cache_key(config, fk_pairs, source="single_phase")
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


def save_pregenerated_cache(
    fk_pairs: List[Tuple[float, float]],
    images: np.ndarray,
    targets: np.ndarray,
    pair_idx: np.ndarray,
    config: dict,
) -> str:
    """Save pre-generated images directly to the standard cache format.

    Used by ``scripts/generate_grid_pairs.py`` to persist the images
    collected during convergence checking, so no separate dataset-generation
    pass is needed before training.

    Args:
        fk_pairs: Converged ``(f, k)`` pairs.
        images: ``(N, 3, H, W)`` float32 array of training images.
        targets: ``(N, 2)`` float32 array of ``[f, k]`` targets.
        pair_idx: ``(N,)`` int64 array mapping each sample to its pair.
        config: Full configuration dict (used for cache dir and key).

    Returns:
        Path to the saved ``.npz`` cache file.
    """
    cache_dir = config["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    key = cache_key(config, fk_pairs, source="grid")
    cache_path = os.path.join(cache_dir, f"dataset_{key}.npz")

    np.savez_compressed(
        cache_path,
        images=images,
        targets=targets,
        pair_idx=pair_idx,
        fk_pairs=np.array(fk_pairs, dtype=np.float32),
    )
    return cache_path
