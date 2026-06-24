# Gray-Scott Inverse Parameter Predictor

An **inverse parameter predictor** for the Gray-Scott reaction-diffusion model.
Given a simulator-generated final-state image (plus initial-condition information),
the model predicts the two reaction parameters that produced it: **`f`** (feed
rate) and **`k`** (kill rate). The diffusion constants `Du = 0.16` and `Dv = 0.08`
are fixed and are not predicted — this is a 2-D regression problem.

## How it works

1. A dense grid of `(f, k)` candidates is checked for convergence by running the
   forward simulator from multiple random seeds. Only pairs that produce a stable,
   patterned steady state (meaningful spatial variance, low temporal drift) are kept.
2. The images collected during convergence checking are saved directly as training
   data — no second simulation pass needed.
3. A CNN (ResNet-18 by default) is trained to map `(3, 64, 64)` images → `(f, k)`.

### Model input

Each sample is a `(3, 64, 64)` float32 tensor:

| Channel | Content |
|---------|---------|
| 0 | `U_final` — final U field |
| 1 | `V_final` — final V field |
| 2 | `V_initial` — seeded initial V field at t=0 |

Each channel is independently min-max normalized to `[-1, 1]`.

### Convergence criteria

A `(f, k)` pair is accepted if a majority of seeds pass both:
- **Variance ≥ 0.005** — V field has spatial structure (not a uniform/dead state)
- **Mean absolute change ≤ 0.005** — V field barely moves between a mid-run
  snapshot and the end (pattern has settled)

## Repository layout

```
config/default.yaml        # all hyperparameters, paths, thresholds
src/gsinverse/
  simulator.py             # Gray-Scott forward simulator (9-point Laplacian, MPS/CUDA)
  convergence.py           # convergence checking and (f, k) grid generation
  datagen.py               # dataset cache builder
  dataset.py               # torch Dataset + train/val split by (f, k) pair
  models/                  # build_model(config) factory: CNN and ViT
  train.py                 # training loop (HuberLoss, AdamW, CosineAnnealingLR)
  evaluate.py              # MAE/R² metrics and sample prediction plots
  utils.py                 # device select, normalization, TargetScaler, config loading
scripts/
  generate_grid_pairs.py   # run convergence grid → save pairs + training cache
  train.py                 # end-to-end training entry point
  evaluate.py              # evaluation entry point
tests/                     # pytest tests for the simulator
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

All commands read defaults from `config/default.yaml`. Override any value
with `--set a.b.c=value` (repeatable).

### 1. Generate converged `(f, k)` pairs and training images

Runs a convergence filter over a dense `(f, k)` grid and saves both the
filtered pairs and the training image cache in one pass.

```bash
python scripts/generate_grid_pairs.py \
    --config config/default.yaml \
    --device mps        # or cuda / cpu
```

Crash-safe: progress is checkpointed every 10 candidates. Re-running the
same command resumes from where it left off.

Outputs:
- `data/fk_pairs_grid.npy` — converged `(f, k)` pairs
- `data/cache/dataset_<hash>.npz` — training image cache

### 2. Train

```bash
python scripts/train.py \
    --config config/default.yaml \
    --fk-pairs data/fk_pairs_grid.npy
```

To train the ViT instead of the default CNN:

```bash
python scripts/train.py \
    --config config/default.yaml \
    --fk-pairs data/fk_pairs_grid.npy \
    --set model.name=vit
```

Key hyperparameters (all overridable via `--set`):

| Key | Default | Notes |
|-----|---------|-------|
| `model.backbone` | `resnet18` | any timm CNN name |
| `model.dropout` | `0.0` | head dropout |
| `train.huber_delta` | `0.1` | HuberLoss delta (targets scaled to [0,1]) |
| `train.early_stopping_patience` | `null` | epochs without improvement before stopping |
| `train.epochs` | `50` | |
| `train.lr` | `1e-4` | AdamW learning rate |

#### Weights & Biases

```bash
python scripts/train.py \
    --config config/default.yaml \
    --fk-pairs data/fk_pairs_grid.npy \
    --set wandb.enabled=true \
    --set wandb.project=gray-scott-inverse
```

Wandb is off by default.

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --config config/default.yaml \
    --fk-pairs data/fk_pairs_grid.npy \
    --checkpoint checkpoints/best_cnn.pt \
    --output-dir checkpoints/eval
```

Produces:
- `metrics.png` — MAE/R² scatter plots for `f` and `k`
- `sample_predictions.png` — 5×3 grid: V_init | Dataset V | Simulated with predicted params

The predicted-params simulation starts from the same `V_initial` as the
dataset sample, so differences are due to `(f, k)` error alone.

## Model options

Set `model.name` in the config or via `--set model.name=...`:

- **`cnn`** (default) — timm CNN backbone (default ResNet-18), global average
  pooling, small MLP head. Change backbone with `--set model.backbone=resnet34`.
- **`vit`** — vit_tiny-sized Vision Transformer, native 64×64 input, patch
  size 8 (64 tokens), explicit regression head.

## Train/val split

The 80/20 split is performed on `(f, k)` pairs, not on individual images.
The same parameter pair never appears in both splits, preventing label leakage
across seeds.

## Tests

```bash
pytest
```

Covers the simulator: Laplacian kernel sums to zero, output shapes, and
determinism under a fixed seed.
