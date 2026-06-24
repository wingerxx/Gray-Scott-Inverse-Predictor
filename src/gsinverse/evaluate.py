"""Evaluation and validation routines for the inverse parameter predictor.

Two routines:

1. :func:`evaluate_and_plot` -- aggregate MAE/R2 and scatter plots.
2. :func:`sample_predictions` -- 5-row grid of V_init / Dataset V / predicted-params simulation.

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
from .utils import TargetScaler, V_INIT_MAX

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
def sample_predictions(
    model: torch.nn.Module,
    val_dataset: Dataset,
    config: dict,
    device: torch.device,
    scaler: Optional[TargetScaler] = None,
    n: int = 5,
    save_path: Optional[str] = None,
) -> list:
    """5×3 grid: V_init | Dataset V | Simulated with predicted params.

    Each row is one validation sample. The predicted-params simulation starts
    from the same V_init as the dataset sample so differences are due to
    (f, k) error alone.

    Args:
        model: Trained regressor.
        val_dataset: Validation dataset.
        config: Full configuration dict.
        device: Device to run inference on.
        scaler: Optional scaler to inverse-transform predictions and targets.
        n: Number of samples to show (rows).
        save_path: If given, save the figure to this path.

    Returns:
        List of dicts with keys ``true_fk`` and ``pred_fk`` for each row.
    """
    model.eval()
    sim_cfg = config["simulator"]

    indices = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))
    results = []

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    # plt.subplots collapses to a 1-D array when n == 1; force 2-D indexing.
    axes = np.atleast_2d(axes)
    col_titles = ["V_init", "Dataset V", "Simulated, predicted params"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        image, target = val_dataset[idx]
        target_orig = _to_original(target.numpy()[None], scaler)[0]
        true_f, true_k = float(target_orig[0]), float(target_orig[1])

        pred = model(image.unsqueeze(0).to(device)).cpu().numpy()
        pred = _to_original(pred, scaler)[0]
        pred_f, pred_k = (max(0.0, float(v)) for v in pred)

        v_init_norm = image[2].numpy()
        v0 = torch.tensor((v_init_norm + 1) / 2 * V_INIT_MAX, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        u0 = torch.ones(1, 1, sim_cfg["size"], sim_cfg["size"])

        _, v_pred = simulate_gray_scott(
            du=sim_cfg["du"], dv=sim_cfg["dv"],
            f=pred_f, k=pred_k,
            iterations=sim_cfg["iterations"],
            size=sim_cfg["size"],
            u0=u0, v0=v0,
            device="cpu",
        )

        axes[row, 0].imshow(image[2].numpy(), cmap="viridis")
        axes[row, 0].set_ylabel(f"true f={true_f:.4f}\nk={true_k:.4f}", fontsize=8)
        axes[row, 1].imshow(image[1].numpy(), cmap="viridis")
        axes[row, 2].imshow(v_pred.numpy(), cmap="viridis")
        axes[row, 2].set_xlabel(f"pred f={pred_f:.4f}, k={pred_k:.4f}", fontsize=8)

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        results.append({"true_fk": (true_f, true_k), "pred_fk": (pred_f, pred_k)})

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return results
