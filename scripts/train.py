#!/usr/bin/env python
"""End-to-end training entry point.

Usage:
    python scripts/train.py --config config/default.yaml \\
        --fk-pairs data/fk_pairs.npy
"""

import argparse
import os

import numpy as np

from gsinverse.dataset import build_train_val_datasets, split_pairs
from gsinverse.train import train
from gsinverse.utils import TargetScaler, load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--fk-pairs",
        default="data/fk_pairs.npy",
        help="Path to the (f, k) pairs produced by extract_params.py",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="Override config: a.b.c=value"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_global_seed(config["seed"])

    fk_pairs = [tuple(p) for p in np.load(args.fk_pairs)]

    # Fit the scaler on training pairs only to avoid val leakage.
    train_idx, _ = split_pairs(fk_pairs, config)
    train_fk = np.array([fk_pairs[i] for i in train_idx], dtype=np.float32)
    scaler = TargetScaler().fit(train_fk)

    # Save scaler alongside the checkpoint so evaluate.py can reload it.
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler_path = os.path.join(checkpoint_dir, f"scaler_{config['model']['name']}.npz")
    scaler.save(scaler_path)
    print(f"Target scaler saved -> {scaler_path}")
    print(f"  f: [{scaler.min_[0]:.4f}, {scaler.min_[0]+scaler.scale_[0]:.4f}]  "
          f"k: [{scaler.min_[1]:.4f}, {scaler.min_[1]+scaler.scale_[1]:.4f}]")

    train_dataset, val_dataset = build_train_val_datasets(fk_pairs, config, scaler=scaler)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    _, checkpoint_path = train(config, train_dataset, val_dataset)
    print(f"Best checkpoint saved -> {checkpoint_path}")


if __name__ == "__main__":
    main()
