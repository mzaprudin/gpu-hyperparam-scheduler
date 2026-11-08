# %%
import argparse
import os
import json


# %%
def print_args(args):
    # Print parsed arguments
    print("\nRunning with parameters:")
    for arg, value in vars(args).items():
        print(f"{arg.replace('_', ' ').capitalize()}: {value}")


# %%
def parse_args(argv=None, *, optimizer=False):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a time-series classifier" if not optimizer else "Run multi-GPU hyperparameter search"
    )

    parser.add_argument("--gpu", type=int, default=0, help="Visible GPU index (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Run training on CPU")
    parser.add_argument("--step", type=int, default=1, help="Stride between training windows")
    parser.add_argument("--model-config", help="JSON file containing ModelParameters")
    if optimizer:
        parser.add_argument("--gpus", help="Comma-separated visible GPU indices, e.g. 0,1")
        parser.add_argument("--min-epochs", type=int, default=3)
        parser.add_argument("--max-epochs", type=int, default=9)
        parser.add_argument("--max-configurations", type=int, default=200)
        parser.add_argument("--initial-points", type=int, default=5)
        parser.add_argument("--genetic-only", action="store_true")
        parser.add_argument("--mp-context", choices=("spawn", "forkserver"), default="spawn")

    parser.add_argument(
        "--model", type=str, default=None, help="Model to load (default: None)"
    )

    parser.add_argument(
        "--window", type=int, default=200, help="Sequence window size (default: 200)"
    )

    parser.add_argument(
        "--forward", type=int, default=75, help="Forward window length (default: 75)"
    )

    parser.add_argument(
        "--skip", type=int, default=15, help="Skip length (default: 15)"
    )

    parser.add_argument(
        "--batch", type=int, default=512, help="Batch size (default: 512)"
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: None)"
    )

    parser.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")

    parser.add_argument(
        "--test-hours",
        type=int,
        default=96,
        help="Number of hours for testing (default: 96)",
    )

    parser.add_argument(
        "--val-hours",
        type=int,
        default=96,
        help="Number of hours for validation (default: 96)",
    )

    parser.add_argument(
        "--train-hours",
        type=int,
        default=336,
        help="Number of hours for training (default: 336)",
    )

    parser.add_argument(
        "--hours-offset",
        type=int,
        default=0,
        help="Hours offset from latest timestamp in history table (default: 0)",
    )

    parser.add_argument(
        "--life-hours",
        type=int,
        default=12,
        help="Min token life hours (default: 12)",
    )

    parser.add_argument(
        "--tp", type=float, default=0.02, help="Min TP ratio (default: 0.02)"
    )

    parser.add_argument(
        "--threshold", type=float, default=0.65, help="Min threshold (default: 0.65)"
    )

    parser.add_argument(
        "--up", type=float, default=0.1, help="Up more than (default: 0.1)"
    )

    parser.add_argument(
        "--dn", type=float, default=0.1, help="Down less than (default: 0.1)"
    )

    parser.add_argument(
        "--range",
        type=float,
        default=0.1,
        help="Min volatility range percent (default: 0.1)",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=300,
        help="Minimum trading volume threshold (default: 300)",
    )

    parser.add_argument(
        "--liquidity",
        type=float,
        default=100,
        help="Minimum sol liquidity in pool (default: 100)",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        default=False,
        help="Reverse filter direction if set (default: False)",
    )

    parser.add_argument(
        "--standard-normalizer",
        action="store_true",
        default=False,
        help="Use standard normalizer instead of MinMaxScaler (default: False)",
    )

    parser.add_argument(
        "--test", action="store_true", default=False, help="Test only (default: False)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration JSON file (default: None)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/demo.pkl",
        help="Path to a trusted local pickle dataset (default: data/demo.pkl)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts (default: artifacts)",
    )

    parser.add_argument(
        "--name", type=str, default=None, help="Setup name (default: None)"
    )

    parser.add_argument(
        "--limit", type=int, default=0, help="Tokens to load (default: 0 - all)"
    )

    # Apply JSON defaults before parsing so explicit CLI options always win.
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            with open(preliminary.config, encoding="utf-8") as stream:
                config = json.load(stream)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read configuration: {exc}")
        if not isinstance(config, dict):
            parser.error("Configuration must be a JSON object")
        known = {action.dest for action in parser._actions}
        unknown = set(config) - known
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for name in ("window", "forward", "step", "batch", "epochs", "test_hours", "val_hours"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.skip < 0 or args.skip >= args.forward:
        parser.error("skip must be non-negative and smaller than forward")
    if optimizer:
        if args.cpu:
            parser.error("GPU search requires a GPU; use gpu-hyperparam-train --cpu for CPU checks")
        if args.min_epochs <= 0 or args.max_epochs < args.min_epochs:
            parser.error("epoch limits must satisfy 0 < min_epochs <= max_epochs")
        if args.max_configurations <= 0 or args.initial_points <= 0:
            parser.error("max_configurations and initial_points must be positive")
    return args
