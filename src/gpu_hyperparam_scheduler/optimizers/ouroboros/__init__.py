"""
Ouroboros Optimizer Package

This package provides a hybrid optimization approach combining gradient descent,
binary search, and historical knowledge integration for hyperparameter optimization.

The main components are:
- OuroborosOptimizer: The main optimizer class
- OuroborosConfig: Configuration for the optimizer
- OuroborosInitConfig: Initialization configuration
"""

from .optimizer import OuroborosOptimizer
from .config import OuroborosConfig, OuroborosInitConfig

__all__ = [
    "OuroborosOptimizer",
    "OuroborosConfig",
    "OuroborosInitConfig",
]
