"""Command-line entry points; argument help does not initialize TensorFlow."""

import json
import logging
import os
from pathlib import Path
import sys

from .args import parse_args


def _configure():
    from .logging import setup_logging
    setup_logging()


def train():
    args = parse_args()
    if args.cpu:
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    from .training_args import create_training_params
    from .create_model import ModelParameters
    from .model_trainer import ModelTrainer
    from .utils import TrainerError

    _configure()
    try:
        params = create_training_params(args)
        if not Path(params.data_file).is_file():
            raise ValueError(f"Dataset not found: {params.data_file}. Run gpu-hyperparam-demo-data first.")
        model_params = None
        if args.model_config:
            with open(args.model_config, encoding="utf-8") as stream:
                model_params = ModelParameters(**json.load(stream))
        trainer = ModelTrainer(params)
        trainer.prepare()
        if params.test_only:
            if not params.model_to_load:
                raise ValueError("--test requires --model")
            trainer.test()
        else:
            trainer.evaluate(model_params)
    except (OSError, ValueError, TypeError, TrainerError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


def optimize():
    args = parse_args(optimizer=True)
    import tensorflow as tf
    from .training_args import create_training_params, copy_training_params
    from .data_loader import load_master_generators
    from .optimizers.ouroboros import OuroborosConfig, OuroborosInitConfig, OuroborosOptimizer

    _configure()
    try:
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            raise ValueError("No GPUs visible to TensorFlow. Use gpu-hyperparam-check --smoke for a CPU check; run search on a GPU machine with uv sync --extra gpu.")
        indices = [int(value) for value in args.gpus.split(",")] if args.gpus else list(range(len(devices)))
        if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(devices) for i in indices):
            raise ValueError(f"GPU indices must be unique and between 0 and {len(devices) - 1}")
        params = create_training_params(args)
        if not Path(params.data_file).is_file():
            raise ValueError(f"Dataset not found: {params.data_file}")
        generators = load_master_generators(params)
        copy_training_params(params, generators.params)
        config = OuroborosConfig(
            min_epochs=args.min_epochs, max_epochs=args.max_epochs,
            max_configurations=args.max_configurations,
            num_initial_points=args.initial_points,
        )
        optimizer = OuroborosOptimizer(OuroborosInitConfig(
            training_params=params, train_master=generators.train,
            val_master=generators.val, test_master=generators.test,
            config=config, gpu_indices=indices, mp_context=args.mp_context,
            genetic_only=args.genetic_only,
        ))
        best = optimizer.optimize()
        logging.getLogger(__name__).info("Best parameters: %s", best)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Search unavailable: {exc}", file=sys.stderr)
        return 1
    return 0
