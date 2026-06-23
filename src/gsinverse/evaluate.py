"""Evaluation and validation routines for the inverse parameter predictor.

Four routines, adapted from the prototype's 4-D (`du, dv, f, k`) versions
down to 2-D (`f, k`), since `du`/`dv` are fixed constants in this project:

1. :func:`evaluate_and_plot` -- aggregate MAE/R2 and scatter plots.
2. :func:`inspect_random_prediction` -- single-sample inference check.
3. :func:`inspect_ground_truth_simulation` -- simulator sanity check.
4. :func:`compare_simulations` -- 3-way shared-seed comparison.

All routines accept an optional ``scaler`` argument. When provided, model
outputs and dataset targets are inverse-transformed back to original physical
units before metrics are computed and parameters are displayed.
"""

import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

from .simulator import simulate_gray_scott
from .utils import TargetScaler

PARAM_NAMES = ["f", "k"]


def _to_original(arr: np.ndarray, scaler: Optional[TargetScaler]) -> np.ndarray:
    """Inverse-transform ``arr`` if a scaler is provided, else return as-is."""
    if scaler is not None:
        return scaler.inverse_transform(arr)
    return arr


@torch.no_grad()
def evaluate_and_plot(
    model: torch.nn.Module,
    val_dataset: Dataset,
    device: torch.device,
    scaler: Optional[TargetScaler] = None,
    batch_size: int = 32,
    save_path: Optional[str] = None,
) -> dict:
    """Compute MAE/R2 for `f` and `k` and plot predicted-vs-true scatter plots.

    Predictions and targets are inverse-transformed to original physical units
    before metrics are computed, so MAE is in the same units as ``f`` and ``k``.

    Args:
        model: Trained regressor.
        val_dataset: Validation dataset (targets may be scaled).
        device: Device to run inference on.
        scaler: Optional :class:`~gsinverse.utils.TargetScaler` used during
            training. When provided, outputs are inverse-transformed before
            plotting and metric computation.
        batch_size: Batch size for inference.
        save_path: If given, save the figure to this path.

    Returns:
        A dict with keys ``mae`` and ``r2``, each mapping ``"f"``/``"k"`` to
        their respective metric values in original units.
    """
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    for images, targets in loader:
        preds = model(images.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    preds = _to_original(np.concatenate(all_preds, axis=0), scaler)
    targets = _to_original(np.concatenate(all_targets, axis=0), scaler)

    metrics = {"mae": {}, "r2": {}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, name in enumerate(PARAM_NAMES):
        y_true = targets[:, i]
        y_pred = preds[:, i]

        metrics["mae"][name] = mean_absolute_error(y_true, y_pred)
        metrics["r2"][name] = r2_score(y_true, y_pred)

        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(
            f"{name}: MAE={metrics['mae'][name]:.5f}, R2={metrics['r2'][name]:.4f}"
        )
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    return metrics


@torch.no_grad()
def inspect_random_prediction(
    model: torch.nn.Module,
    val_dataset: Dataset,
    config: dict,
    device: torch.device,
    scaler: Optional[TargetScaler] = None,
    save_path: Optional[str] = None,
) -> dict:
    """Run the model on a random validation sample and re-simulate the prediction.

    Displays the real V-channel (from the dataset sample) next to the
    V-channel of a fresh simulation run with the model's predicted ``(f, k)``.

    Args:
        model: Trained regressor.
        val_dataset: Validation dataset.
        config: Full configuration dict.
        device: Device to run inference on.
        scaler: Optional scaler to inverse-transform predictions and targets.
        save_path: If given, save the figure to this path.

    Returns:
        A dict with keys ``true_fk`` and ``pred_fk`` in original units.
    """
    model.eval()
    sim_cfg = config["simulator"]

    idx = random.randrange(len(val_dataset))
    image, target = val_dataset[idx]

    pred = model(image.unsqueeze(0).to(device)).cpu().numpy()
    pred = _to_original(pred, scaler)[0]
    target_orig = _to_original(target.numpy()[None], scaler)[0]

    pred_f, pred_k = (max(0.0, float(v)) for v in pred)
    true_f, true_k = float(target_orig[0]), float(target_orig[1])

    seed = random.randint(0, 2**31 - 1)
    _, v_pred, _, _ = simulate_gray_scott(
        du=sim_cfg["du"],
        dv=sim_cfg["dv"],
        f=pred_f,
        k=pred_k,
        iterations=sim_cfg["iterations"],
        size=sim_cfg["size"],
        patch_radius=sim_cfg["patch_radius"],
        patch_prob=sim_cfg["patch_prob"],
        seed=seed,
        device="cpu",
        return_initial=True,
    )

    real_v = image[1].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_v, cmap="viridis")
    axes[0].set_title(f"Real V (true f={true_f:.4f}, k={true_k:.4f})")
    axes[1].imshow(v_pred.numpy(), cmap="viridis")
    axes[1].set_title(f"Predicted V (pred f={pred_f:.4f}, k={pred_k:.4f})")
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    return {"true_fk": (true_f, true_k), "pred_fk": (pred_f, pred_k)}


def inspect_ground_truth_simulation(
    val_dataset: Dataset,
    config: dict,
    scaler: Optional[TargetScaler] = None,
    idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> dict:
    """Re-simulate using a sample's true ``(f, k)`` and compare to the dataset image.

    Sanity-checks the simulator against a generated sample. Since the
    training data and the simulator are the same system, the fresh
    simulation should produce a visually similar steady-state pattern.

    Args:
        val_dataset: Validation dataset.
        config: Full configuration dict.
        scaler: Optional scaler to inverse-transform targets to original units.
        idx: Index of the sample to inspect. If ``None``, a random index is used.
        save_path: If given, save the figure to this path.

    Returns:
        A dict with key ``fk`` giving the ``(f, k)`` pair used in original units.
    """
    sim_cfg = config["simulator"]

    if idx is None:
        idx = random.randrange(len(val_dataset))
    image, target = val_dataset[idx]
    target_orig = _to_original(target.numpy()[None], scaler)[0]
    f, k = float(target_orig[0]), float(target_orig[1])

    seed = random.randint(0, 2**31 - 1)
    _, v_sim, _, _ = simulate_gray_scott(
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

    dataset_v = image[1].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(dataset_v, cmap="viridis")
    axes[0].set_title(f"Dataset V (f={f:.4f}, k={k:.4f})")
    axes[1].imshow(v_sim.numpy(), cmap="viridis")
    axes[1].set_title("Fresh simulation V (same f, k)")
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    return {"fk": (f, k)}


@torch.no_grad()
def compare_simulations(
    model: torch.nn.Module,
    val_dataset: Dataset,
    config: dict,
    device: torch.device,
    scaler: Optional[TargetScaler] = None,
    idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> dict:
    """3-way comparison: dataset sample, true-param simulation, predicted-param simulation.

    Both simulations use a *shared* random seed so differences in the
    resulting patterns are attributable only to the ``(f, k)`` values, not
    to differing initial conditions. ``Du``/``Dv`` are kept fixed.
    Predictions are clamped to ``max(0, ...)`` before simulating to avoid
    numerical blow-ups.

    Args:
        model: Trained regressor.
        val_dataset: Validation dataset.
        config: Full configuration dict.
        device: Device to run inference on.
        scaler: Optional scaler to inverse-transform predictions and targets.
        idx: Index of the sample to inspect. If ``None``, a random index is used.
        save_path: If given, save the figure to this path.

    Returns:
        A dict with keys ``true_fk`` and ``pred_fk`` in original units.
    """
    model.eval()
    sim_cfg = config["simulator"]

    if idx is None:
        idx = random.randrange(len(val_dataset))
    image, target = val_dataset[idx]
    target_orig = _to_original(target.numpy()[None], scaler)[0]
    true_f, true_k = float(target_orig[0]), float(target_orig[1])

    pred = model(image.unsqueeze(0).to(device)).cpu().numpy()
    pred = _to_original(pred, scaler)[0]
    pred_f, pred_k = (max(0.0, float(v)) for v in pred)

    shared_seed = random.randint(0, 2**31 - 1)

    def _run(f, k):
        _, v, _, _ = simulate_gray_scott(
            du=sim_cfg["du"],
            dv=sim_cfg["dv"],
            f=f,
            k=k,
            iterations=sim_cfg["iterations"],
            size=sim_cfg["size"],
            patch_radius=sim_cfg["patch_radius"],
            patch_prob=sim_cfg["patch_prob"],
            seed=shared_seed,
            device="cpu",
            return_initial=True,
        )
        return v.numpy()

    v_true = _run(true_f, true_k)
    v_pred = _run(pred_f, pred_k)
    dataset_v = image[1].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(dataset_v, cmap="viridis")
    axes[0].set_title(f"Dataset V\n(f={true_f:.4f}, k={true_k:.4f})")
    axes[1].imshow(v_true, cmap="viridis")
    axes[1].set_title(f"Simulated, true params\n(f={true_f:.4f}, k={true_k:.4f})")
    axes[2].imshow(v_pred, cmap="viridis")
    axes[2].set_title(f"Simulated, predicted params\n(f={pred_f:.4f}, k={pred_k:.4f})")
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    return {"true_fk": (true_f, true_k), "pred_fk": (pred_f, pred_k)}
