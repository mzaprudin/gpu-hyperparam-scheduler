"""
Data structures and configurations for optimization.
"""

import multiprocessing as mp
from multiprocessing.queues import Queue
from typing import Generic, TypeVar, Optional
from enum import Enum
from dataclasses import dataclass, field
import time
import math
from typing import List, Optional

import numpy as np

from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.create_model import ModelParameters
from gpu_hyperparam_scheduler.data_generator import MasterGenerator

T = TypeVar("T")

class EvaluationMode(Enum):
    """Mode for threshold evaluation."""
    EVALUATE = "evaluate"  # Find best threshold where tp ratio > min_tp_ratio
    TEST = "test"         # Use min_threshold without optimization


class TypedQueue(Queue, Generic[T]):
    """Type-hinted queue for better code readability."""

    def __init__(self, ctx):
        super().__init__(ctx=ctx)


@dataclass
class ASHAConfig:
    """Configuration for Asynchronous Successive Halving Algorithm."""

    max_epochs: int = 81  # Maximum resources (epochs) per configuration
    reduction_factor: int = 3  # Reduction factor for successive halving
    min_epochs: int = 1  # Minimum resources for first evaluation
    max_configurations: int = 100  # Maximum total configurations to try
    mutation_rate: float = 0.2  # Rate of mutation for genetic operations
    use_adaptive_ranges: bool = False  # Whether to use adaptive parameter ranges

    def __post_init__(self):
        # Validate configuration
        if not math.log(self.max_epochs, self.reduction_factor).is_integer():
            raise ValueError(
                f"max_epochs ({self.max_epochs}) must be a power of reduction_factor ({self.reduction_factor})"
            )


@dataclass
class EvaluationResult:
    """Complete result of a model evaluation."""

    params: ModelParameters
    precision: float
    epochs: int
    config_id: str
    timestamp: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class ConfigEvaluation:
    """Tracks evaluation of a single configuration."""

    params: ModelParameters
    current_epochs: int = 1
    precision: float = float("-inf")
    completed: bool = False
    bracket: int = 0  # Tracks which ASHA bracket this belongs to
    results: List[EvaluationResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)


@dataclass
class OptimizationTask:
    """Task definition for parallel optimization."""

    params: ModelParameters
    epochs: int
    config_id: str
    gpu_idx: int  # Added to track which GPU this task is assigned to
    start_time: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Result from optimization task."""

    params: ModelParameters
    precision: float
    threshold: float
    tp_ratio: float
    epochs: int
    config_id: str
    start_time: float
    end_time: float = field(default_factory=time.time)


@dataclass
class WorkerConfig:
    """Configuration for GPU worker process."""

    task_queue: TypedQueue[Optional[OptimizationTask]]
    result_queue: TypedQueue[Optional[OptimizationResult]]
    base_params: TrainingParameters
    train_master: MasterGenerator
    val_master: MasterGenerator
    test_master: MasterGenerator
    gpu_idx: int = 0
    stop_event: Optional[mp.Event] = None
    health_queue: Optional[TypedQueue[float]] = None


@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation."""

    total_samples: int
    class_one_total: int
    certain_predictions: int
    uncertain_count: int
    tp_count: int
    fn_count: int
    fp_count: int
    uncertain_class_one: int
    accuracy: float
    recall: float
    f1_score: float
    precision: float


@dataclass
class OptimalThresholdResult:
    """Results from optimal threshold calculation."""

    optimal_idx: int
    optimal_threshold: float
    optimal_precision: float
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    tp_ratio: np.ndarray  # Interpolated TP ratio for all thresholds
