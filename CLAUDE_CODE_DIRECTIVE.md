# Directive: Build the Gray-Scott Inverse Parameter Predictor

Convert my existing prototype (a single Jupyter notebook) into a clean, well-structured, reproducible GitHub Python project. Below is the full design specification. Build the codebase from scratch following this spec — do **not** simply copy the notebook structure. Use proper modules, configs, and CLI entry points.

---

## 1. Project Goal

Build an **inverse parameter predictor** for the Gray-Scott reaction-diffusion model.

- **Input:** a simulator-generated final-state image plus its seed/initial-condition information.
- **Output:** the two reaction parameters **`f` (feed)** and **`k` (kill)**.
- **`Du` and `Dv` are fixed constants** (`Du = 0.16`, `Dv = 0.08`) across all data and are **not predicted**. The model is a 2-D regressor only.

### Key design principle: self-consistent data
The forward simulator is used as **both the training-data generator and the validation tool**, so the image→parameter mapping the model learns is internally consistent. Training images are NOT taken from the original HDF5 dataset. We only borrow the **(f, k) parameter pairs** from the HDF5 dataset (those pairs are known to lie in regions of parameter space that converge to stable steady-state patterns).

---

## 2. Data Pipeline

### 2a. Extract parameters from the HDF5 dataset
The original dataset is an HDF5 file (path will be supplied via config; example: `grayscott_64x64_1-1000.hdf5`, ~1000 records).

- Record keys are top-level groups: `record_0`, `record_1`, ... (use `list(f.keys())`).
- Each record has a `meta` group whose `.attrs` contain: `du`, `dv`, `feed`, `kill`.
- **Extract only `feed` and `kill`** from each record to build the master list of `(f, k)` pairs. Ignore the stored images entirely. Optionally assert that `du ≈ 0.16` and `dv ≈ 0.08` to validate the assumption.

### 2b. Generate training images via the forward simulator
For each `(f, k)` pair, run the forward simulator (Section 3) to produce a final-state image.

- Use a **randomly generated seed** for each generated sample (seeds are NOT stored in or tied to the HDF5 records).
- Support **multiple seeds per `(f, k)` pair** (configurable `seeds_per_param`, default e.g. 5) to expand the dataset beyond ~1000 samples and force the model to learn **seed-invariant** features characteristic of `f` and `k`.
- Run each simulation for **1000 iterations** (matches the convergence horizon of the original dataset).

### 2c. Model input tensor
Each sample is a **`(3, 64, 64)`** float tensor, native resolution (no upscaling):
1. `U_final` — final U field
2. `V_final` — final V field
3. `V_initial` — the seeded initial V field at t=0 (encodes seed/initial-condition info explicitly so the model can reason about what changed)

Apply **per-channel, per-sample min-max normalization to `[-1, 1]`** (same scheme as the prototype: `2*(x - min)/(max - min + 1e-8) - 1`). Normalize each of the 3 channels independently.

> Note: the prototype upscaled 64×64 → 224×224 for a ViT. **Do not upscale.** Keep native 64×64.

### 2d. Targets
`[f, k]` only — a 2-D regression target per sample.

### 2e. Dataset generation strategy
Implement **pre-generation with on-disk caching** as the default (generating 1000 iterations per sample on-the-fly every epoch is too slow):
- A `generate_dataset` script builds all `(image, [f,k])` samples once and caches them (e.g. to a compressed `.npz`/HDF5/`.pt` cache keyed by a config hash).
- Training loads from this cache.
- Also expose an optional **on-the-fly mode** (regenerate with fresh random seeds each epoch) behind a config flag, for users who want maximal augmentation and have the compute.
- 80/20 train/val split with a fixed seed (`random_state=42`). Split on the **`(f, k)` pairs**, not on generated images, so the same parameter pair never appears in both train and val (prevents leakage when using multiple seeds per pair).

---

## 3. Forward Simulator (port verbatim, then modularize)

Port these two functions from the notebook into a `simulator` module. The math is verified correct — preserve it exactly.

### 3a. `create_initial_fields(grid_length, patch_radius, patch_prob, seed=None, device='cpu')`
- `u = ones(1,1,L,L)`, `v = zeros(1,1,L,L)`.
- RNG: `torch.Generator(device)`; `manual_seed(seed)` if seed given, else `rng.seed()`.
- Multi-patch seeding: `stride = patch_radius*2 + 1`; loop `x,y` over the interior grid on that stride; with probability `patch_prob`, set the `(2*patch_radius+1)` square patch to `u=0.50`, `v=0.25`.
- Symmetry-breaking noise: `u += rand(u.shape)*0.05`, `v += rand(v.shape)*0.05` (same generator).
- Returns `(u, v)`.

### 3b. `simulate_gray_scott(du, dv, f, k, iterations=1000, size=64, patch_radius=2, patch_prob=0.5, seed=None, device='cpu')`
- Build initial fields via `create_initial_fields`.
- Discrete Laplacian kernel (isotropic 9-point, sums to zero):
  ```
  [[0.05, 0.20, 0.05],
   [0.20, -1.00, 0.20],
   [0.05, 0.20, 0.05]]
  ```
- `dt = 1.0` (stable for Du=0.16, Dv=0.08).
- Per step: `uvv = u*v*v`; circular-pad U and V (`F.pad(..., mode='circular')`); `lap = F.conv2d(pad, kernel)`; then
  ```
  u = u + dt*(du*lap_u - uvv + f*(1 - u))
  v = v + dt*(dv*lap_v + uvv - (f + k)*v)
  ```
- **Change the default `iterations` to 1000** (the prototype used 1500; the real dataset converges by ~1000).
- Also expose a variant/option that returns the **initial V field** alongside the final fields, since the data pipeline needs `V_initial` as the 3rd input channel. (Either return it, or have `create_initial_fields` results captured by the caller — your choice, but the dataset builder must be able to obtain `V_initial`.)
- Hardcode/clamp `Du=0.16`, `Dv=0.08` at the data-generation layer, but keep `simulate_gray_scott` general (accepts du/dv) so it stays reusable.

---

## 4. Model

Make the architecture **swappable via config** (`model.name`), because we want to empirically compare two candidates on native 64×64 input. Both take `in_chans=3`, output 2 values (`f`, `k`).

### Candidate A — CNN backbone (recommended default)
A compact CNN regressor: e.g. ResNet-18 or EfficientNet-B0 backbone adapted to `in_chans=3`, global average pooling, then a small MLP head → 2 outputs. CNNs suit the local texture patterns of Gray-Scott at 64×64 and are more data-efficient.

### Candidate B — Small-input ViT
A ViT configured for native 64×64 with **patch size 8** (→ 8×8 = 64 tokens, a meaningful sequence length without upscaling). Use `timm` with `in_chans=3`, `num_classes=2`, and an appropriate small-image config. (The prototype's `vit_tiny_patch16_224` on upscaled input is the wrong fit — replace it.)

Provide both behind a common factory (`build_model(config)`), so swapping is a one-line config change.

---

## 5. Training

Port the training loop and keep these settings (config-driven):
- **Loss:** `nn.HuberLoss` (robust to outliers).
- **Optimizer:** `AdamW`, `lr=1e-4`, `weight_decay=1e-4`.
- **Scheduler:** `CosineAnnealingLR`, `T_max=epochs`.
- **Epochs:** 50 (default, configurable).
- **Batch size:** 32.
- **Checkpointing:** save the best model by lowest validation loss to a checkpoints dir; reload best weights at the end.
- Log per-epoch train/val loss (console + optionally a CSV/TensorBoard).
- **Device auto-selection:** `cuda` → `mps` (Apple Silicon) → `cpu`.

---

## 6. Evaluation & Validation

Port all four validation routines, adapted from 4-D to **2-D (`f`, `k`)**:

1. **Aggregate metrics** (`evaluate_and_plot`): MAE and R² per parameter across the full validation set; scatter plots of predicted vs. true with a perfect-prediction diagonal — now **2 plots** (`f`, `k`), not 4.
2. **Single-sample inference check** (`inspect_random_prediction`): run the model on a random val sample, re-simulate using the predicted `(f, k)`, display real vs. predicted V-channel.
3. **Ground-truth simulation check** (`inspect_ground_truth_simulation`): re-simulate using true `(f, k)` to sanity-check the simulator against a generated sample. (Should match closely now that training data and simulator are the same system.)
4. **3-way comparison** (`compare_simulations`): use a **shared seed** for both true-param and predicted-param simulations for a fair 1-to-1 visual comparison. Keep `Du`/`Dv` fixed; only `f`, `k` vary. Clamp predictions with `max(0, ...)` before simulating to avoid numerical blow-ups.

---

## 7. Repository Structure

Create a clean, installable Python project. Suggested layout (adapt sensibly):

```
gray-scott-inverse/
├── README.md                  # overview, install, usage, design summary
├── requirements.txt
├── pyproject.toml             # installable package "gsinverse"
├── .gitignore                 # ignore data/, checkpoints/, caches, __pycache__, .venv
├── config/
│   └── default.yaml           # all hyperparams, paths, model.name, seeds_per_param, etc.
├── src/gsinverse/
│   ├── __init__.py
│   ├── simulator.py           # create_initial_fields, simulate_gray_scott
│   ├── params.py              # extract (f,k) pairs from HDF5
│   ├── datagen.py             # build & cache the (image, [f,k]) dataset
│   ├── dataset.py             # torch Dataset over the cache (+ optional on-the-fly mode)
│   ├── models/
│   │   ├── __init__.py        # build_model(config) factory
│   │   ├── cnn.py
│   │   └── vit.py
│   ├── train.py               # training loop
│   ├── evaluate.py            # the 4 validation routines + plotting
│   └── utils.py               # device select, normalization, seeding helpers
├── scripts/
│   ├── extract_params.py      # HDF5 -> (f,k) list
│   ├── generate_dataset.py    # (f,k) -> cached images
│   ├── train.py               # end-to-end train entry point
│   └── evaluate.py            # load checkpoint, run validations
├── tests/
│   └── test_simulator.py      # shape checks, kernel sums to 0, determinism under fixed seed
└── notebooks/
    └── exploration.ipynb      # optional, for interactive inspection
```

### Requirements
Pin: `torch`, `timm`, `h5py`, `numpy`, `matplotlib`, `scikit-learn`, `pyyaml` (and `tqdm` for progress bars). Include both CNN (torchvision or timm) and ViT (timm) dependencies.

### Engineering expectations
- All hyperparameters and paths come from `config/default.yaml` (overridable via CLI args). No hardcoded absolute paths like `/Users/...`.
- Type hints and docstrings on public functions.
- A reproducible seed control for the train/val split and (optionally) global RNG.
- A `README.md` that explains the project goal, the self-consistent-data design, how to run each stage (extract → generate → train → evaluate), and the model-swap config option.
- Basic `pytest` tests for the simulator (kernel sums to ~0, output shapes, determinism under a fixed seed).
- Add a `.gitignore` and make sure generated data, caches, and checkpoints are excluded from version control.

---

## 8. Migration notes / things to change from the prototype

- **Drop Du/Dv as targets** → 2-D regression (`f`, `k`).
- **Stop using HDF5 images** for training; use HDF5 **only** for `(f, k)` pairs.
- **Generate training images** from the forward simulator with random seeds, 1000 iterations, multiple seeds per `(f, k)`.
- **Add `V_initial` as a 3rd input channel** → input is `(3, 64, 64)`.
- **No upscaling** — native 64×64 throughout.
- **Replace the ViT-tiny-224** model with a config-swappable CNN (default) and small-input ViT (patch8).
- **Change simulator default iterations to 1000.**
- **Split on `(f, k)` pairs**, not on generated images, to avoid leakage across seeds.

When done, print a short summary of what was built and the exact commands to run the full pipeline.
