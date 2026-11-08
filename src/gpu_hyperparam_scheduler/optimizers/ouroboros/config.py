"""Configuration classes for Ouroboros Optimizer."""

from dataclasses import dataclass
from typing import List, Optional

from gpu_hyperparam_scheduler.data_generator import MasterGenerator
from gpu_hyperparam_scheduler.config import TrainingParameters

# Constants for timeouts and health checks
RESULT_TIMEOUT = 12 * 3600  # 12 hours timeout for getting results
WORKER_JOIN_TIMEOUT = 3600  # 1 hour timeout for worker cleanup
WORKER_TERMINATE_TIMEOUT = 60  # 1 minute timeout for worker termination
HEALTH_CHECK_INTERVAL = 300  # 5 minutes between health checks
HEALTH_CHECK_TIMEOUT = 10  # 10 seconds timeout for health check response
TASK_TIMEOUT = 24 * 3600  # 24 hours timeout for individual tasks
DEFAULT_TASK_DURATION = 300  # 5 minutes default expected duration

# Constants for optimization process
TOP_CONFIGS_TO_KEEP = 10  # Number of best configurations to maintain
TOP_REGION_PERCENTILE = 0.2  # Top 20% of performance regions to consider promising
DURATION_DECAY = 0.7  # Weight for historical duration in moving average
DURATION_UPDATE = 0.3  # Weight for new duration in moving average
MAX_SENSITIVE_PARAMS = 3  # Number of most sensitive parameters to optimize
GRADIENT_STEP_RATIO = 0.1  # Fraction of range to use as step size
TASK_DURATION_THRESHOLD = 2.0  # Multiplier for expected duration before warning


@dataclass
class OuroborosConfig:
    """Configuration for Ouroboros Optimizer."""

    # Core parameters
    min_epochs: int = 10
    max_epochs: int = 100
    max_configurations: int = 100

    # Exploration parameters
    num_initial_points: int = 5
    gradient_points_per_param: int = 3
    binary_search_depth: int = 4

    # Adaptation parameters
    exploration_ratio: float = 0.3  # Ratio of configurations for exploration vs exploitation
    min_improvement_threshold: float = 0.01  # Minimum improvement to consider a direction promising
    mutation_rate: float = 0.2  # Rate of mutation for genetic algorithm operations


@dataclass
class OuroborosInitConfig:
    """Configuration for OuroborosOptimizer initialization."""

    training_params: TrainingParameters
    train_master: MasterGenerator
    val_master: MasterGenerator
    test_master: MasterGenerator
    config: Optional[OuroborosConfig] = None
    gpu_indices: Optional[List[int]] = None
    mp_context: str = "spawn"
    genetic_only: bool = False
    """If True, only use genetic algorithm (initial and mutation tasks)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gpu_indices is None:
            self.gpu_indices = [0]
        if not self.gpu_indices or any(i < 0 for i in self.gpu_indices):
            raise ValueError("gpu_indices must contain non-negative GPU indices")
        if not isinstance(self.train_master, MasterGenerator):
            raise ValueError("train_master must be a MasterGenerator instance")
        if not isinstance(self.val_master, MasterGenerator):
            raise ValueError("val_master must be a MasterGenerator instance")
        if not isinstance(self.test_master, MasterGenerator):
            raise ValueError("test_master must be a MasterGenerator instance")
