#!/usr/bin/env python
"""Load a trained checkpoint and run the validation routines.

Usage:
    python scripts/evaluate.py --config config/default.yaml \\
        --fk-pairs data/fk_pairs.npy \\
        --checkpoint checkpoints/best_cnn.pt
"""

import argparse
import os

import numpy as np
import torch

from gsinverse.dataset import build_train_val_datasets
from gsinverse.evaluate import (
    compare_simulations,
    evaluate_and_plot,
    inspect_ground_truth_simulation,
    inspect_random_prediction,
)
from gsinverse.models import build_model
from gsinverse.utils import get_device, load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--fk-pairs",
        default="data/fk_pairs.npy",
        help="Path to the (f, k) pairs produced by extract_params.py",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt)")
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

    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fk_pairs = [tuple(p) for p in np.load(args.fk_pairs)]
    _, val_dataset = build_train_val_datasets(fk_pairs, config, show_progress=False)

    metrics = evaluate_and_plot(
        model, val_dataset, device, save_path=os.path.join(args.output_dir, "metrics.png")
    )
    print("Validation metrics:", metrics)

    inspect_random_prediction(
        model, val_dataset, config, device,
        save_path=os.path.join(args.output_dir, "random_prediction.png"),
    )
    inspect_ground_truth_simulation(
        val_dataset, config,
        save_path=os.path.join(args.output_dir, "ground_truth_simulation.png"),
    )
    compare_simulations(
        model, val_dataset, config, device,
        save_path=os.path.join(args.output_dir, "compare_simulations.png"),
    )

    print(f"Evaluation plots saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
