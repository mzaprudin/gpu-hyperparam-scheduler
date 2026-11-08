# Source and migration

Source: [pepelax/neurosol](https://github.com/pepelax/neurosol), branch `main`,
commit [`e37cb145f8a7d6ef28880d8b4024db3d2df5987d`](https://github.com/pepelax/neurosol/commit/e37cb145f8a7d6ef28880d8b4024db3d2df5987d).
The relevant model, trainer, GPU-worker and optimizer files are identical in the
tips of `main`, `community` and `val` at extraction time.

The latest optimizer / worker changes were authored on 4 February 2025
(`9083e33` in `main`, equivalent to `8d5a232` in `community`). The latest original
launcher changes were authored on 13 February (`c75b68b` / `443120d`). Several
committer timestamps in `main` are 19 February because the commits were copied.

`SOURCE.json` records each extracted file's original path and SHA-256 digest
before migration. This repository starts a new Git history; the original history
remains accessible through the source link.

## Included

- CNN–LSTM–Transformer architecture, custom focal loss and normalization.
- Training, validation threshold selection, evaluation and visualizations.
- Local-data preparation and window generators.
- Modular Ouroboros optimizer, parameter ranges, history, scheduler and workers.
- Command-line launchers and the callable `run_optimization` helper.

The Solana RPC client, wallet community detector, QuestDB ingestion, stair
prediction API and older monolithic / alternative optimizer implementations
were omitted. Local training does not require those services.

## Changes made during extraction

- Moved imports from `src.*` to the installable `gpu_hyperparam_scheduler` package.
- Added `pyproject.toml`, `uv.lock`, optional CUDA dependencies, console commands,
  synthetic examples and CPU CI. The TensorFlow / Keras stack stays on the
  original 2.16 / 3.3 series.
- Replaced machine-specific paths with project-relative defaults and added an
  explicit CPU training mode. Full GPU search still requires GPU hardware.
- Made CLI values override configuration defaults, validated config keys, and
  added controls for search budgets and visible GPU indices.
- Fixed the missing liquidity argument in the raw-data loader, bounded data
  preprocessing pools by the input size, and allowed single-core hosts.
- Made custom-loss serialization accept Keras configuration fields, specified
  the direction of precision-monitoring callbacks and created output directories.
- Fixed restored results being removed from nonexistent active-task entries,
  pending-trial duplicate detection, trial metadata and epoch-budget persistence.
- Fixed the controller staying in initial exploration, references to missing
  genetic-mode / random-generator attributes, gradient-result argument handling,
  available-trial budget accounting and cleanup after partial initialization.
- Switched worker startup to `spawn`. Worker failures now stop the controller
  instead of silently waiting for another result. Search state is saved atomically
  within the experiment's artifact directory.

## Verification and research limits

CPU verification covers imports of every module, a real model forward/backward
pass, serialization round trip, full one-epoch training and checkpoint evaluation.
Scheduler tests use the real task/result queues with a lightweight evaluator to
exercise saving and resuming without allocating GPU devices.

Multi-GPU training, CUDA compatibility, speedup and fault tolerance on actual GPU
hardware have not been measured here. The retained search strategies are
experimental; no advantage over standard HPO libraries is claimed.

The historical trainer returns test-split precision to the optimizer. That
evaluation behavior is retained in this extraction. For an unbiased final quality
estimate, use a further untouched dataset; scores used during search are not an
independent benchmark. The synthetic example verifies execution, not market
prediction quality.
