#!/usr/bin/env python
"""Generate a convergence-filtered grid of (f, k) pairs and training images.

Produces a dense uniform grid of candidate (f, k) pairs, runs each through
the forward simulator, and keeps only those that converge to a stable,
patterned steady state.

The simulations run during convergence checking are reused directly as
training images -- no second simulation pass is needed. Both the filtered
pairs (.npy) and the training image cache (.npz) are saved in one pass, so
training can run directly afterward with scripts/train.py.

Crash recovery: progress is checkpointed as work proceeds. Image arrays are
appended to a resizable HDF5 file (float32, no re-serialization), and a small
JSON sidecar records the counters and converged pairs (written atomically).
If the process dies, re-running with the same arguments resumes from where it
left off.

Usage:
    python scripts/generate_grid_pairs.py --config config/default.yaml
    python scripts/generate_grid_pairs.py --config config/default.yaml --device mps
"""

import argparse
import json
import os

import h5py
import numpy as np

from gsinverse.convergence import check_pair_convergence, generate_fk_grid
from gsinverse.datagen import save_pregenerated_cache
from gsinverse.utils import load_config

# Number of candidates between checkpoint writes.
CHECKPOINT_EVERY = 10


def _append_samples_h5(h5_path, images, targets, pair_idx):
    """Append new samples to resizable HDF5 datasets, creating them if needed.

    Images and targets are stored as float32 (so resumed runs never silently
    upcast to float64); pair_idx as int64. Appends only the rows passed in --
    no whole-file rewrite.
    """
    if len(images) == 0:
        return
    images = np.asarray(images, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    pair_idx = np.asarray(pair_idx, dtype=np.int64)

    with h5py.File(h5_path, "a") as hf:
        if "images" not in hf:
            hf.create_dataset(
                "images", data=images,
                maxshape=(None,) + images.shape[1:], chunks=True, dtype="float32",
            )
            hf.create_dataset(
                "targets", data=targets,
                maxshape=(None, targets.shape[1]), chunks=True, dtype="float32",
            )
            hf.create_dataset(
                "pair_idx", data=pair_idx,
                maxshape=(None,), chunks=True, dtype="int64",
            )
        else:
            for name, arr in (("images", images), ("targets", targets), ("pair_idx", pair_idx)):
                ds = hf[name]
                n_old = ds.shape[0]
                ds.resize(n_old + arr.shape[0], axis=0)
                ds[n_old:] = arr


def _write_meta_atomic(meta_path, meta):
    """Write the JSON sidecar atomically (temp file + os.replace)."""
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, meta_path)


def _load_checkpoint(meta_path, h5_path):
    """Restore checkpoint state as float32 arrays.

    The JSON sidecar is the source of truth for the committed sample count.
    If the HDF5 file holds more rows than the sidecar recorded (a crash
    between the HDF5 append and the sidecar write), the extra rows are
    truncated so resume reproduces exactly an uninterrupted run.
    """
    with open(meta_path) as fh:
        meta = json.load(fh)
    n_samples = meta.get("n_samples", 0)

    images, targets, pair_idx = [], [], []
    if n_samples > 0 and os.path.exists(h5_path):
        with h5py.File(h5_path, "a") as hf:
            if "images" in hf:
                for name in ("images", "targets", "pair_idx"):
                    if hf[name].shape[0] > n_samples:
                        hf[name].resize(n_samples, axis=0)
                images = list(np.asarray(hf["images"][:n_samples], dtype=np.float32))
                targets = list(np.asarray(hf["targets"][:n_samples], dtype=np.float32))
                pair_idx = list(np.asarray(hf["pair_idx"][:n_samples], dtype=np.int64))
    return meta, images, targets, pair_idx


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
    max_stability = conv_cfg.get("max_stability", 0.005)

    # Crash-recovery checkpoint: small JSON sidecar (counters, converged pairs)
    # plus a resizable HDF5 file holding the accumulated image arrays.
    meta_path = output_path.replace(".npy", "_checkpoint.json")
    h5_path = output_path.replace(".npy", "_checkpoint.h5")

    if os.path.exists(meta_path):
        meta, all_images, all_targets, all_pair_idx = _load_checkpoint(meta_path, h5_path)
        start_idx = meta["next_idx"]
        converged = [tuple(p) for p in meta["converged"]]
        dead_count = meta["dead"]
        unstable_count = meta["unstable"]
        print(f"Resuming from checkpoint at candidate {start_idx}/{len(candidates)} "
              f"({len(converged)} converged, {len(all_images)} images so far)")
    else:
        start_idx = 0
        converged = []
        dead_count = 0
        unstable_count = 0
        all_images, all_targets, all_pair_idx = [], [], []

    # Number of samples already persisted to the HDF5 checkpoint.
    n_flushed = len(all_images)

    iterator = range(start_idx, len(candidates))
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(candidates), initial=start_idx,
                        desc="Checking convergence")
    except ImportError:
        pass

    for pair_i in iterator:
        f, k = candidates[pair_i]

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

        # Checkpoint periodically: append any new images to HDF5 first, then
        # commit the sidecar atomically (sidecar is the source of truth).
        if (pair_i + 1) % CHECKPOINT_EVERY == 0:
            _append_samples_h5(
                h5_path,
                all_images[n_flushed:],
                all_targets[n_flushed:],
                all_pair_idx[n_flushed:],
            )
            n_flushed = len(all_images)
            _write_meta_atomic(meta_path, {
                "next_idx": pair_i + 1,
                "converged": [list(p) for p in converged],
                "dead": dead_count,
                "unstable": unstable_count,
                "n_samples": n_flushed,
            })

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
    images_arr = np.stack(all_images, axis=0).astype(np.float32)
    targets_arr = np.array(all_targets, dtype=np.float32)
    pair_idx_arr = np.array(all_pair_idx, dtype=np.int64)
    cache_path = save_pregenerated_cache(converged_f32, images_arr, targets_arr, pair_idx_arr, config)
    print(f"  Training images:  {len(images_arr)} samples ({images_arr.shape})")
    print(f"\nTraining cache saved -> {cache_path}")

    # Remove checkpoint files now that we have a clean result.
    for path in (meta_path, h5_path):
        if os.path.exists(path):
            os.remove(path)

    print("\nReady to train -- run train.py directly:")
    print(f"  python scripts/train.py --fk-pairs {output_path}")


if __name__ == "__main__":
    main()
