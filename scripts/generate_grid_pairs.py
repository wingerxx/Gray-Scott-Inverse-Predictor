#!/usr/bin/env python
"""Generate a convergence-filtered grid of (f, k) pairs and training images.

Produces a dense uniform grid of candidate (f, k) pairs, runs each through
the forward simulator, and keeps only those that converge to a stable,
patterned steady state.

The simulations run during convergence checking are reused directly as
training images -- no second simulation pass is needed. Both the filtered
pairs (.npy) and the training image cache (.npz) are saved in one pass,
so generate_dataset.py can be skipped entirely.

Crash recovery: progress is checkpointed to a .json file after each
candidate is checked. If the process dies, re-running with the same
arguments resumes from where it left off.

Usage:
    python scripts/generate_grid_pairs.py --config config/default.yaml
    python scripts/generate_grid_pairs.py --config config/default.yaml --device mps
"""

import argparse
import json
import os

import numpy as np

from gsinverse.convergence import check_pair_convergence, generate_fk_grid
from gsinverse.datagen import save_pregenerated_cache
from gsinverse.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--output", default=None,
        help="Output path for pairs .npy file. Defaults to grid.output in config.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device to use for simulations (e.g. 'cpu', 'mps', 'cuda'). "
             "Defaults to CPU.",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="Override config: a.b.c=value"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    grid_cfg = config["grid"]
    sim_cfg = config["simulator"]
    conv_cfg = grid_cfg.get("convergence", {})
    output_path = args.output or grid_cfg["output"]
    device = args.device or "cpu"

    candidates = generate_fk_grid(
        f_min=grid_cfg["f_min"],
        f_max=grid_cfg["f_max"],
        f_steps=grid_cfg["f_steps"],
        k_min=grid_cfg["k_min"],
        k_max=grid_cfg["k_max"],
        k_steps=grid_cfg["k_steps"],
    )
    print(f"Generated {len(candidates)} candidate (f, k) pairs "
          f"({grid_cfg['f_steps']} f-values × {grid_cfg['k_steps']} k-values)")
    print(f"Running convergence check at {sim_cfg['iterations']} iterations "
          f"+ {conv_cfg.get('stability_steps', 200)} stability steps "
          f"({conv_cfg.get('n_seeds', 3)} seeds each) on device={device!r} ...\n")

    iterations = conv_cfg.get("iterations", 2000)
    stability_steps = conv_cfg.get("stability_steps", 200)
    n_seeds = conv_cfg.get("n_seeds", 3)
    min_variance = conv_cfg.get("min_variance", 0.005)
    max_stability = conv_cfg.get("max_stability", 0.001)

    # Checkpoint file lets us resume after a crash.
    checkpoint_file = output_path.replace(".npy", "_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as fh:
            ckpt = json.load(fh)
        start_idx = ckpt["next_idx"]
        converged = [tuple(p) for p in ckpt["converged"]]
        dead_count = ckpt["dead"]
        unstable_count = ckpt["unstable"]
        all_images = [np.array(img) for img in ckpt["images"]]
        all_targets = [np.array(t) for t in ckpt["targets"]]
        all_pair_idx = list(ckpt["pair_idx"])
        print(f"Resuming from checkpoint at candidate {start_idx}/{len(candidates)} "
              f"({len(converged)} converged so far)")
    else:
        start_idx = 0
        converged = []
        dead_count = 0
        unstable_count = 0
        all_images = []
        all_targets = []
        all_pair_idx = []

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(candidates), total=len(candidates), initial=start_idx, desc="Checking convergence")
    except ImportError:
        iterator = enumerate(candidates)

    for pair_i, (f, k) in iterator:
        if pair_i < start_idx:
            continue

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
            pair_out_idx = len(converged)
            converged.append((f, k))
            for img in samples:
                all_images.append(img)
                all_targets.append([f, k])
                all_pair_idx.append(pair_out_idx)
        elif "dead" in reason:
            dead_count += 1
        else:
            unstable_count += 1

        # Write checkpoint every 10 candidates.
        if (pair_i + 1) % 10 == 0:
            ckpt = {
                "next_idx": pair_i + 1,
                "converged": [list(p) for p in converged],
                "dead": dead_count,
                "unstable": unstable_count,
                "images": [img.tolist() for img in all_images],
                "targets": [t if isinstance(t, list) else list(t) for t in all_targets],
                "pair_idx": all_pair_idx,
            }
            with open(checkpoint_file, "w") as fh:
                json.dump(ckpt, fh)

    total = len(candidates)
    print(f"\nConvergence results:")
    print(f"  Converged (kept): {len(converged)} / {total} "
          f"({100*len(converged)/total:.1f}%)")
    print(f"  Dead/uniform:     {dead_count}  ({100*dead_count/total:.1f}%)")
    print(f"  Unstable:         {unstable_count} ({100*unstable_count/total:.1f}%)")

    if not converged:
        print("\nNo pairs converged — try relaxing thresholds via --set grid.convergence.*")
        return

    # Save filtered pairs.
    fk_array = np.array(converged, dtype=np.float32)
    np.save(output_path, fk_array)
    # Round to float32 so the cache key matches what train.py computes after
    # loading the .npy file (which stores float32).
    converged_f32 = [(float(f), float(k)) for f, k in fk_array]
    print(f"\nSaved {len(converged)} converged pairs -> {output_path}")
    print(f"  f range: [{fk_array[:,0].min():.4f}, {fk_array[:,0].max():.4f}]")
    print(f"  k range: [{fk_array[:,1].min():.4f}, {fk_array[:,1].max():.4f}]")

    # Save pre-generated training images directly to the standard cache.
    images_arr = np.stack(all_images, axis=0)
    targets_arr = np.array(all_targets, dtype=np.float32)
    pair_idx_arr = np.array(all_pair_idx, dtype=np.int64)
    cache_path = save_pregenerated_cache(converged_f32, images_arr, targets_arr, pair_idx_arr, config)
    print(f"  Training images:  {len(images_arr)} samples ({images_arr.shape})")
    print(f"\nTraining cache saved -> {cache_path}")

    # Remove checkpoint file now that we have a clean result.
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("\nSkip generate_dataset.py -- run train.py directly:")
    print(f"  python scripts/train.py --fk-pairs {output_path}")


if __name__ == "__main__":
    main()
