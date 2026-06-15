#!/usr/bin/env python
"""Generate and cache the (image, [f, k]) training dataset.

Usage:
    python scripts/generate_dataset.py --config config/default.yaml \\
        --fk-pairs data/fk_pairs.npy
"""

import argparse

import numpy as np

from gsinverse.datagen import generate_and_cache
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
    cache_path = generate_and_cache(fk_pairs, config)
    print(f"Dataset cached -> {cache_path}")


if __name__ == "__main__":
    main()
