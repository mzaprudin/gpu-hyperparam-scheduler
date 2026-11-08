# GPU Hyperparam Scheduler

English · [Русский](README.RU.md)

Train time-series classifiers in TensorFlow/Keras and search their hyperparameters
across GPUs. A controller schedules independent trials asynchronously, saves their
results and continues an existing search when relaunched.

Python 3.12; the TensorFlow 2.16 / Keras 3.3 environment is pinned in `uv.lock`.
The included example generates synthetic time series locally.

## Engineering decisions

- **Training pipeline:** window generators feed a CNN–Transformer–BiLSTM model with
  a custom focal loss that gives extra weight to false positives.
- **GPU scheduling:** one process per GPU takes trials from a shared task queue.
  The controller processes results as workers finish.
- **Resumable search:** saves parameters, scores, thresholds, epoch budgets and
  durations. Relaunching adds evaluations to the saved history.
- **Configuration tracking:** checks completed and pending trials for duplicates.
  Search combines random exploration, sensitivity probes, parameter-range probes and
  genetic mutations.

The [architecture guide](ARCHITECTURE.md) explains the model, worker lifecycle,
search strategies and how training, threshold selection and evaluation fit together.

## Run the CPU example

From the repository root:

```bash
uv sync --locked
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-train --config examples/cpu.json
```

This runs data preparation, window generation, one training epoch, checkpoint
reload and evaluation. Outputs include a `.keras` checkpoint, JSON report and plot.
The [example guide](examples/README.md) documents the recorded CPU run and links
its report and plot, with input columns and options for using another dataset.

## Search across GPUs

On Linux with NVIDIA GPUs and compatible drivers, use the generated data above:

```bash
uv sync --locked --extra gpu
uv run gpu-hyperparam-scheduler --config examples/search.json --gpus 0,1
```

Repeat the command to continue the same search with additional evaluations.
The example guide covers device selection, search budgets and result files.

## Verification

[CPU CI](.github/workflows/ci.yml) checks package imports, a real training step,
checkpoint save/reload, full example training and scheduler save/resume behavior,
then builds the package. The [tests](tests/) exercise the CLI and search controller.

[Source history and migration notes](PROVENANCE.md).
