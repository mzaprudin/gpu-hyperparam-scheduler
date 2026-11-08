# GPU Hyperparam Scheduler

Time-series classification with a CNN–LSTM–Transformer model and asynchronous
hyperparameter search across multiple GPUs. Extracted from Neurosol; see
[PROVENANCE.md](PROVENANCE.md) for the source revision and migration notes.

## Setup

```bash
uv sync --locked
uv run gpu-hyperparam-check
uv run gpu-hyperparam-check --smoke
uv run pytest
```

Python 3.12 is used to preserve the original TensorFlow 2.16 / Keras 3 training
stack. `uv.lock` records the resolved versions. The default install supports CPU
checks without downloading the optional NVIDIA CUDA libraries.

## CPU example

```bash
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-train --config examples/cpu.json
```

The example generates synthetic market-shaped time series locally. It exercises
loading, labeling, splitting, model training, checkpoint reload and evaluation;
its metrics are not evidence of predictive quality on real market data.

## Multi-GPU search

On a Linux machine with compatible NVIDIA GPUs and drivers:

```bash
uv sync --locked --extra gpu
uv run gpu-hyperparam-check
uv run gpu-hyperparam-scheduler --config examples/search.json --gpus 0,1
```

Use `--data` to supply a local trusted pickle dataset. See
[examples/README.md](examples/README.md) for the expected columns and options.
GPU search exits with an actionable message if no GPUs are visible. GPU execution
and scaling have not been validated in the CPU-only extraction environment.

The scheduler evaluates independent configurations in one worker process per
GPU. Search combines random exploration, parameter sensitivity probes, interval
refinement and genetic mutations. Results are saved and can be resumed.

## Code

- `src/gpu_hyperparam_scheduler/create_model.py`: model, custom loss and normalization.
- `src/gpu_hyperparam_scheduler/model_trainer.py`: training, checkpoints and evaluation.
- `src/gpu_hyperparam_scheduler/data_*.py`: preparation and windowed batch generation.
- `src/gpu_hyperparam_scheduler/optimizers/ouroboros/`: search, task management, history
  and GPU workers.
- `tests/`: CPU integration checks and optimizer regression tests.

```bash
uv run gpu-hyperparam-train --help
uv run gpu-hyperparam-scheduler --help
uv run ruff check .
uv build
```

Training artifacts, virtual environments and private datasets are ignored by Git.
