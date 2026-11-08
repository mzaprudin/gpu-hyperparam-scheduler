"""
Parameter sampling and validation for hyperparameter optimization.
"""

from dataclasses import asdict
from gpu_hyperparam_scheduler.create_model import ModelParameters
from gpu_hyperparam_scheduler.hyperparameters import ModelHyperparameters


class ParameterSampler:
    """Handles sampling of model parameters within defined ranges."""

    def __init__(self, hyperparameters: ModelHyperparameters):
        """
        Initialize sampler with hyperparameter ranges.

        Args:
            hyperparameters: ModelHyperparameters instance defining parameter ranges
        """
        self.hp = hyperparameters
        # Set standard_normalizer from base params if provided
        if hasattr(hyperparameters, "standard_normalizer"):
            self.hp.set_standard_normalizer(hyperparameters.standard_normalizer)

    def sample_parameters(self) -> ModelParameters:
        """Sample a random valid parameter configuration."""
        params_dict = self.hp.random_params()
        return ModelParameters(**params_dict)

    def validate_parameters(self, params: ModelParameters) -> bool:
        """Validate parameter configuration against constraints."""
        try:
            # Validate parameter ranges
            param_dict = asdict(params)
            for name, value in param_dict.items():
                if name == "mlp_units":
                    # Validate MLP units are decreasing
                    if not all(value[i] > value[i + 1] for i in range(len(value) - 1)):
                        return False
                elif name in ["num_heads", "head_size", "ff_dim"]:
                    # Validate power of 2 parameters
                    if not value & (value - 1) == 0:  # Check if power of 2
                        return False
            return True
        except (AttributeError, TypeError, IndexError):
            return False
