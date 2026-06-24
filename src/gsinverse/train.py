"""Training loop for the Gray-Scott inverse parameter predictor."""

import csv
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .models import build_model
from .utils import get_device


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    """Run one epoch of training or evaluation.

    If ``optimizer`` is provided, the model is put in training mode and
    weights are updated. Otherwise, the model is evaluated with gradients
    disabled.

    Returns:
        The mean loss over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = criterion(preds, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train(
    config: dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[nn.Module, str]:
    """Train a model according to ``config`` and checkpoint the best weights.

    Args:
        config: Full configuration dict.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.

    Returns:
        A tuple ``(model, checkpoint_path)`` where ``model`` has the best
        validation-loss weights loaded, and ``checkpoint_path`` is the path
        to the saved checkpoint.
    """
    train_cfg = config["train"]
    device = get_device()

    model = build_model(config).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    huber_delta = train_cfg.get("huber_delta", 0.1)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    epochs = train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{config['model']['name']}.pt")

    log_csv = train_cfg.get("log_csv")
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        with open(log_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "val_loss"])

    wandb_cfg = config.get("wandb", {})
    wandb_run = None
    if wandb_cfg.get("enabled"):
        import wandb

        wandb_run = wandb.init(
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config=config,
        )

    best_val_loss = float("inf")
    early_stop_patience = train_cfg.get("early_stopping_patience", None)
    epochs_without_improvement = 0

    try:
        for epoch in range(1, epochs + 1):
            train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = _run_epoch(model, val_loader, criterion, device)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

            if log_csv:
                with open(log_csv, "a", newline="") as fh:
                    csv.writer(fh).writerow([epoch, train_loss, val_loss])

            if wandb_run is not None:
                wandb_run.log(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                    break

        if wandb_run is not None:
            wandb_run.log({"best_val_loss": best_val_loss})
            wandb_run.save(checkpoint_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, checkpoint_path
