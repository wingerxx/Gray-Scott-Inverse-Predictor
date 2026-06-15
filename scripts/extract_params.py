#!/usr/bin/env python
"""Extract (f, k) parameter pairs from the source HDF5 dataset and save them.

Usage:
    python scripts/extract_params.py --config config/default.yaml \\
        --output data/fk_pairs.npy
"""

import argparse

import numpy as np

from gsinverse.params import extract_fk_pairs
from gsinverse.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--output", default="data/fk_pairs.npy", help="Path to save the extracted (f, k) pairs"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="Override config: a.b.c=value"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    fk_pairs = extract_fk_pairs(config["paths"]["hdf5_path"])

    np.save(args.output, np.array(fk_pairs, dtype=np.float32))
    print(f"Extracted {len(fk_pairs)} (f, k) pairs -> {args.output}")


if __name__ == "__main__":
    main()
