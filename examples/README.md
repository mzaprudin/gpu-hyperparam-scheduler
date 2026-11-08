# Examples and input data

Generate a small deterministic dataset and train one epoch on CPU:

```bash
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-train --config examples/cpu.json
```

Outputs are written to `artifacts/cpu-demo/`: a `.keras` checkpoint, a JSON report
and a plot. Preprocessed generators are cached next to the dataset as
`data/demo_mg.pkl`. The raw data remain unchanged.

Evaluate the saved checkpoint:

```bash
uv run gpu-hyperparam-train --config examples/cpu.json --test --model artifacts/cpu-demo/demo.keras
```

Explicit command-line values override JSON settings. JSON keys use argument
destination names (`save_dir`, `val_hours`, `min_epochs`, etc.). Relative paths
are resolved from the current working directory; run examples from the repository
root. Unknown keys and missing configuration files produce an error.

## Data schema

The raw input is a pickle containing `dict[str, pandas.DataFrame]`, one chronological
series per identifier. Each frame needs these numeric columns:

| Column | Meaning |
| --- | --- |
| `block_number` | Monotonically increasing observation index |
| `unix_timestamp` | Observation time in Unix seconds |
| `transactions`, `accounts` | Activity counts |
| `buy_volume`, `sell_volume` | Traded volumes |
| `price` | Positive price |
| `liquidity_token`, `liquidity_sol` | Liquidity features |

The loader computes activity filters and labels, splits by timestamp and creates
window generators. Each split must contain enough usable windows for a batch;
the balanced training split must include both classes. `--data` accepts either
raw data or cached generators created by this package. Historical Neurosol
generator pickles refer to the old `src.*` module paths: regenerate those caches
from the raw DataFrames after migration.

Only load pickle files you trust, because pickle deserialization executes Python.
No real market data or trained weights are distributed with this repository.

## GPU search

```bash
uv sync --locked --extra gpu
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-scheduler --config examples/search.json --gpus 0,1
```

`--gpus` uses indices in TensorFlow's visible device list. For example,
`CUDA_VISIBLE_DEVICES=2,3` together with `--gpus 0,1` selects physical devices
2 and 3. Each worker uses its own device and evaluates independent trials.
Workers use the `spawn` multiprocessing context to avoid inheriting an initialized
TensorFlow runtime.

`artifacts/search/optimization_results.json` stores the search history. Repeating
the command resumes it and evaluates up to `--max-configurations` additional
configurations. Use a separate `--save-dir` for each dataset or experiment so
unrelated results are not mixed. `--initial-points`, `--min-epochs`, `--max-epochs`
and `--genetic-only` control the search budget and strategy.

The CUDA extra follows [TensorFlow's Linux installation guidance](https://www.tensorflow.org/install/pip).
The NVIDIA driver must be installed on the GPU host separately. These GPU commands
are documented for use on GPU hardware; this extraction was validated on CPU.
