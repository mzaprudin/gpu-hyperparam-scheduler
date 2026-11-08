"""Generate deterministic synthetic data with the original input schema."""

import argparse
from pathlib import Path
import pickle


def make_data(series=3, rows=720, seed=42):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    result = {}
    index = np.arange(rows)
    for number in range(series):
        phase = index / 48 + number
        price = np.exp(0.08 * np.sin(phase) + rng.normal(0, 0.002, rows))
        result[f"synthetic-{number}"] = pd.DataFrame({
            "block_number": index,
            "unix_timestamp": 1_735_689_600 + index * 60,
            "transactions": rng.integers(10, 50, rows),
            "accounts": rng.integers(5, 25, rows),
            "buy_volume": rng.uniform(10, 50, rows),
            "sell_volume": rng.uniform(10, 50, rows),
            "price": price,
            "liquidity_token": 10_000 / price,
            "liquidity_sol": rng.uniform(1_000, 1_500, rows),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(make_data(seed=args.seed), stream)
    print(f"Synthetic dataset written to {args.output}")
