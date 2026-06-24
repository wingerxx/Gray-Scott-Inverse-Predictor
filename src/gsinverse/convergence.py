"""Convergence checking and grid generation for the Gray-Scott parameter space.

Rather than relying on the source HDF5 dataset (which contains only 5 unique
(f, k) pairs), this module generates a dense grid of candidate pairs and
filters them by whether the simulator produces a stable, patterned steady
state.

A pair is considered converged if:
  1. The V field has meaningful spatial variance (pattern exists, not uniform).
  2. The V field changes little between a mid-run snapshot and the end of the
     run (pattern has stopped evolving).

Crucially, the simulations run during convergence checking are reused
directly as training images -- no second simulation pass is needed.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from .simulator import simulate_gray_scott_two_phase
from .utils import normalize_minmax

# Convergence criteria defaults -- exposed so callers can override.
DEFAULT_MIN_VARIANCE = 0.005
DEFAULT_MAX_STABILITY = 0.001


def _build_sample(
    u_snapshot: torch.Tensor,
    v_snapshot: torch.Tensor,
    v_initial: torch.Tensor,
) -> np.ndarray:
    """Build a normalized ``(3, 64, 64)`` float32 input tensor from raw fields."""
    channels = [
        normalize_minmax(u_snapshot.cpu().numpy()),
        normalize_minmax(v_snapshot.cpu().numpy()),
        normalize_minmax(v_initial.cpu().numpy()),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def check_pair_convergence(
    f: float,
    k: float,
    du: float = 0.16,
    dv: float = 0.08,
    iterations: int = 2000,
    stability_steps: int = 200,
    size: int = 64,
    patch_radius: int = 2,
    patch_prob: float = 0.5,
    n_seeds: int = 3,
    pair_index: int = 0,
    min_variance: float = DEFAULT_MIN_VARIANCE,
    max_stability: float = DEFAULT_MAX_STABILITY,
    device: str = "cpu",
    return_samples: bool = False,
) -> Tuple:
    """Check whether a ``(f, k)`` pair produces a stable patterned state.

    Runs the simulation from ``n_seeds`` different random initial conditions
    and requires that a majority pass both the variance and stability checks.

    When ``return_samples=True``, the normalized ``(3, H, W)`` images from
    all *passing* seeds are returned alongside the verdict, so they can be
    used directly as training samples without re-simulating.

    Args:
        f: Feed rate.
        k: Kill rate.
        du: Diffusion rate for U.
        dv: Diffusion rate for V.
        iterations: Steps before the stability snapshot.
        stability_steps: Additional steps after the snapshot.
        size: Grid side length.
        patch_radius: Patch radius for initial seeding.
        patch_prob: Patch probability for initial seeding.
        n_seeds: Number of seeds to test per pair.
        pair_index: Index of this pair in the candidate list. Used to derive
            unique, reproducible seeds (``pair_index * n_seeds + s``) so
            every pair gets different initial conditions.
        min_variance: Minimum V-field variance (below = dead/uniform).
        max_stability: Maximum mean absolute change (above = still evolving).
        device: Torch device.
        return_samples: If ``True``, return passing seed images alongside
            the verdict.

    Returns:
        If ``return_samples`` is ``False``: ``(passed, reason)``.
        If ``return_samples`` is ``True``: ``(passed, reason, samples)``
        where ``samples`` is a list of ``(3, H, W)`` float32 arrays from
        passing seeds (empty list if the pair failed).
    """
    passed_seeds = []
    failed_variance = []
    failed_stability = []

    for s in range(n_seeds):
        seed = pair_index * n_seeds + s
        u_snap, v_snap, v_final, v_initial = simulate_gray_scott_two_phase(
            du=du, dv=dv, f=f, k=k,
            iterations=iterations,
            stability_steps=stability_steps,
            size=size,
            patch_radius=patch_radius,
            patch_prob=patch_prob,
            seed=seed,
            device=device,
        )
        variance = v_snap.var().item()
        stability = torch.abs(v_final - v_snap).mean().item()

        if variance >= min_variance and stability <= max_stability:
            if return_samples:
                passed_seeds.append(_build_sample(u_snap, v_snap, v_initial))
            else:
                passed_seeds.append(True)
        elif variance < min_variance:
            failed_variance.append(variance)
        else:
            failed_stability.append(stability)

    majority = n_seeds // 2 + 1
    passed = len(passed_seeds) >= majority

    if passed:
        reason = "converged"
    elif failed_variance:
        reason = f"dead (var={np.mean(failed_variance):.5f})"
    else:
        reason = f"unstable (stability={np.mean(failed_stability):.5f})"

    if return_samples:
        return passed, reason, passed_seeds
    return passed, reason


def generate_fk_grid(
    f_min: float = 0.01,
    f_max: float = 0.09,
    f_steps: int = 30,
    k_min: float = 0.04,
    k_max: float = 0.07,
    k_steps: int = 30,
) -> List[Tuple[float, float]]:
    """Generate a dense uniform grid of ``(f, k)`` candidate pairs.

    Args:
        f_min: Minimum feed rate.
        f_max: Maximum feed rate.
        f_steps: Number of f values.
        k_min: Minimum kill rate.
        k_max: Maximum kill rate.
        k_steps: Number of k values.

    Returns:
        A list of ``(f, k)`` tuples covering the full grid.
    """
    f_vals = np.linspace(f_min, f_max, f_steps)
    k_vals = np.linspace(k_min, k_max, k_steps)
    return [(float(f), float(k)) for f in f_vals for k in k_vals]


def filter_grid_by_convergence(
    candidates: List[Tuple[float, float]],
    config: dict,
    show_progress: bool = True,
    device: str = "cpu",
) -> Tuple[List[Tuple[float, float]], dict, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Filter candidate pairs, collecting training images in the same pass.

    Each passing pair contributes its convergence-check images directly as
    training samples -- no second simulation is needed.

    Args:
        candidates: List of ``(f, k)`` candidate pairs.
        config: Full configuration dict.
        show_progress: If ``True``, display a tqdm progress bar.

    Returns:
        A tuple ``(converged_pairs, stats, images, targets, pair_idx)`` where:
        - ``converged_pairs``: list of ``(f, k)`` tuples that passed.
        - ``stats``: dict with counts of converged/dead/unstable pairs.
        - ``images``: ``(N, 3, H, W)`` float32 array of training images.
        - ``targets``: ``(N, 2)`` float32 array of ``[f, k]`` targets.
        - ``pair_idx``: ``(N,)`` int64 array mapping each sample to its pair.
    """
    sim_cfg = config["simulator"]
    grid_cfg = config.get("grid", {})
    conv_cfg = grid_cfg.get("convergence", {})

    iterations = conv_cfg.get("iterations", 2000)
    stability_steps = conv_cfg.get("stability_steps", 200)
    n_seeds = conv_cfg.get("n_seeds", 3)
    min_variance = conv_cfg.get("min_variance", DEFAULT_MIN_VARIANCE)
    max_stability = conv_cfg.get("max_stability", DEFAULT_MAX_STABILITY)

    iterator = candidates
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(candidates, desc="Checking convergence")

    converged_pairs = []
    dead, unstable = [], []
    all_images, all_targets, all_pair_idx = [], [], []

    for pair_i, (f, k) in enumerate(iterator):
        passed, reason, samples = check_pair_convergence(
            f=f, k=k,
            du=sim_cfg["du"],
            dv=sim_cfg["dv"],
            iterations=iterations,
            stability_steps=stability_steps,
            size=sim_cfg["size"],
            patch_radius=sim_cfg["patch_radius"],
            patch_prob=sim_cfg["patch_prob"],
            n_seeds=n_seeds,
            pair_index=pair_i,
            min_variance=min_variance,
            max_stability=max_stability,
            device=device,
            return_samples=True,
        )

        if passed:
            pair_out_idx = len(converged_pairs)
            converged_pairs.append((f, k))
            for img in samples:
                all_images.append(img)
                all_targets.append([f, k])
                all_pair_idx.append(pair_out_idx)
        elif "dead" in reason:
            dead.append((f, k))
        else:
            unstable.append((f, k))

    stats = {
        "total": len(candidates),
        "converged": len(converged_pairs),
        "dead": len(dead),
        "unstable": len(unstable),
    }

    if not all_images:
        return converged_pairs, stats, None, None, None

    images = np.stack(all_images, axis=0)
    targets = np.array(all_targets, dtype=np.float32)
    pair_idx = np.array(all_pair_idx, dtype=np.int64)

    return converged_pairs, stats, images, targets, pair_idx
