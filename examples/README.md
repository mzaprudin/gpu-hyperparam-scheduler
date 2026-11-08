# Examples and input data

[Русская версия](README.RU.md) · [Project overview](../README.md) ·
[Architecture](../ARCHITECTURE.md)

The CPU example trains a time-series classifier and produces a reloadable Keras
checkpoint, a JSON evaluation report and a probability-distribution plot.
Run all commands from the project root; data and configuration paths are relative
to the current working directory.

## Train on CPU

Use Python 3.12 and the locked dependencies:

```bash
uv sync --locked
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-train --config examples/cpu.json
```

The generator writes `data/demo.pkl`: three synthetic series with 720 rows each,
one observation per minute, using seed `42`. The CPU configuration also sets
seed `42`; it trains one epoch with 16-row windows and a small model.

Training fits on the earlier data, selects a confidence threshold on validation
data, then evaluates test data with that threshold. The console compares
validation and test metrics. Search uses precision from the test report to rank
configurations. See [Architecture](../ARCHITECTURE.md) for the full flow.

| Local output | Contents |
| --- | --- |
| `artifacts/cpu-demo/demo.keras` | Trained model with custom layers and loss |
| `artifacts/cpu-demo/demo.json` | Test metrics, model and training parameters |
| `artifacts/cpu-demo/demo.png` | Probability distribution and threshold analysis |
| `data/demo_mg.pkl` | Prepared train, validation and test generators |

The raw dataset stays at `data/demo.pkl`. The published
[CPU report](results/cpu-demo.json) and [probability plot](../docs/assets/cpu-demo.png)
come from a completed run of this example.

## Evaluate the saved checkpoint

Load the checkpoint and evaluate test data at an explicit threshold of `0.5`:

```bash
uv run gpu-hyperparam-train --config examples/cpu.json \
  --test --model artifacts/cpu-demo/demo.keras \
  --threshold 0.5 --save-dir artifacts/cpu-evaluation
```

This writes `demo.json` and `demo.png` into `artifacts/cpu-evaluation/`, preserving
the training run's report and plot. In `--test` mode, `--threshold` sets the
evaluation threshold directly. Use `results.optimal_threshold` from the original
JSON report to evaluate with the threshold selected during training.

Explicit CLI values override JSON settings. Configuration keys use argument
names with underscores, such as `save_dir`, `val_hours` and `min_epochs`.

## Supply local data

Use `--data path/to/data.pkl` for a trusted pickle containing
`dict[str, pandas.DataFrame]`: one chronological series per identifier.
Each frame needs these numeric columns:

| Column | Meaning |
| --- | --- |
| `block_number` | Monotonically increasing observation index |
| `unix_timestamp` | Observation time in Unix seconds |
| `transactions`, `accounts` | Activity counts |
| `buy_volume`, `sell_volume` | Traded volumes |
| `price` | Positive price |
| `liquidity_token`, `liquidity_sol` | Liquidity features |

Only load pickle files from a trusted source: deserialization executes Python.
Each split needs enough usable windows for a batch; the balanced training split
needs both classes. `--data` also accepts generator caches made by this package.
Regenerate historical Neurosol caches from raw frames to use the current package
paths instead of the former `src.*` imports.

## Search on GPUs and resume

Use Linux with compatible NVIDIA GPUs and an installed NVIDIA driver. Install
the optional CUDA dependencies, check visible devices, then start a search:

```bash
uv sync --locked --extra gpu
uv run gpu-hyperparam-check
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-scheduler --config examples/search.json --gpus 0,1
```

`--gpus` indexes TensorFlow's visible device list. With
`CUDA_VISIBLE_DEVICES=2,3`, `--gpus 0,1` selects physical GPUs 2 and 3.
Each GPU has a worker process started with `spawn`.

History is saved to `artifacts/search/optimization_results.json`. Continue that
search with a budget of up to five additional configurations:

```bash
uv run gpu-hyperparam-scheduler --config examples/search.json \
  --gpus 0,1 --max-configurations 5
```

Keep the same dataset and settings when resuming; use a separate `--save-dir`
for each experiment. `--initial-points`, `--min-epochs`, `--max-epochs` and
`--genetic-only` control the search budget and strategy.
