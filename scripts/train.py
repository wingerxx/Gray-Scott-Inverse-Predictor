#!/usr/bin/env python
"""End-to-end training entry point.

Usage:
    python scripts/train.py --config config/default.yaml \\
        --fk-pairs data/fk_pairs.npy
"""

import argparse

import numpy as np

from gsinverse.dataset import build_train_val_datasets
from gsinverse.train import train
from gsinverse.utils import load_config, set_global_seed


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
    train_dataset, val_dataset = build_train_val_datasets(fk_pairs, config)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    _, checkpoint_path = train(config, train_dataset, val_dataset)
    print(f"Best checkpoint saved -> {checkpoint_path}")


if __name__ == "__main__":
    main()
