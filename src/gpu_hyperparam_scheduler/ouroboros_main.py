"""
Main entry point for the Ouroboros Optimizer.

This module provides a clean interface to initialize and run the optimizer
with appropriate configuration and data generators.
"""

import multiprocessing as mp

from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.data_generator import MasterGenerator
from gpu_hyperparam_scheduler.optimizers.ouroboros import (
    OuroborosOptimizer,
    OuroborosConfig,
    OuroborosInitConfig,
)


def run_optimization(
    base_params: TrainingParameters,
    train_master: MasterGenerator,
    val_master: MasterGenerator,
    test_master: MasterGenerator,
    gpu_indices=None,
    mp_context="fork",
):
    """
    Run Ouroboros optimization process.

    Args:
        base_params: Base training parameters
        train_master: Training data generator
        val_master: Validation data generator
        test_master: Test data generator
        gpu_indices: List of GPU indices to use (default: [0])
        mp_context: Multiprocessing context (default: "fork")

    Returns:
        Tuple of (best parameters, best precision)
    """
    # Create optimizer configuration
    config = OuroborosConfig(
        min_epochs=10,
        max_epochs=100,
        max_configurations=100,
        num_initial_points=5,
        gradient_points_per_param=3,
        binary_search_depth=4,
        exploration_ratio=0.3,
        min_improvement_threshold=0.01,
    )

    # Create initialization configuration
    init_config = OuroborosInitConfig(
        training_params=base_params,
        train_master=train_master,
        val_master=val_master,
        test_master=test_master,
        config=config,
        gpu_indices=gpu_indices,
        mp_context=mp_context,
    )

    # Initialize and run optimizer
    optimizer = OuroborosOptimizer(init_config)
    best_params = optimizer.optimize()

    return best_params, optimizer.best_precision


if __name__ == "__main__":
    from .cli import optimize
    raise SystemExit(optimize())
