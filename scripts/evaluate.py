#!/usr/bin/env python
"""Load a trained checkpoint and run the validation routines.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_cnn.pt
"""

import argparse
import os

import numpy as np
import torch

from gsinverse.dataset import build_train_val_datasets
from gsinverse.evaluate import (
    evaluate_and_plot,
    sample_predictions,
)
from gsinverse.models import build_model
from gsinverse.utils import TargetScaler, get_device, load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--fk-pairs",
        default="data/fk_pairs_grid.npy",
        help="Path to the (f, k) pairs .npy file",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt)")
    parser.add_argument(
        "--scaler",
        default=None,
        help="Path to scaler .npz file. Defaults to scaler_<model>.npz next to the checkpoint.",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/eval", help="Directory to save evaluation plots"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="Override config: a.b.c=value"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_global_seed(config["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scaler — default path mirrors what train.py saves.
    scaler_path = args.scaler or os.path.join(
        os.path.dirname(args.checkpoint),
        f"scaler_{config['model']['name']}.npz",
    )
    scaler = None
    if os.path.exists(scaler_path):
        scaler = TargetScaler.load(scaler_path)
        print(f"Loaded target scaler from {scaler_path}")
    else:
        print(f"No scaler found at {scaler_path} — metrics will be in scaled space.")

    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fk_pairs = [tuple(p) for p in np.load(args.fk_pairs)]
    _, val_dataset = build_train_val_datasets(fk_pairs, config, scaler=scaler, show_progress=False)

    metrics = evaluate_and_plot(
        model, val_dataset, device, scaler=scaler,
        save_path=os.path.join(args.output_dir, "metrics.png"),
    )
    print("Validation metrics:", metrics)

    sample_predictions(
        model, val_dataset, config, device, scaler=scaler,
        save_path=os.path.join(args.output_dir, "sample_predictions.png"),
    )

    print(f"Evaluation plots saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
