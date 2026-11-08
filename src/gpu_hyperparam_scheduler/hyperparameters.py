"""
Hyperparameter management for neural network optimization.
"""

from typing import Dict, Any, List
import numpy as np
from gpu_hyperparam_scheduler.param_range import ParamRange, ParamType


class ModelHyperparameters:
    """Defines and manages the ranges of hyperparameters for model optimization."""

    def __init__(self):
        # Model architecture parameters
        self.param_ranges = {
            # Model architecture parameters - power of 2 values
            "num_heads": ParamRange(2, 12, 2, ParamType.INTEGER),  # [2,4,8,16]
            "head_size": ParamRange(16, 96, 16, ParamType.INTEGER),  # [8,16,32,64,128]
            "ff_dim": ParamRange(64, 256, 32, ParamType.INTEGER),  # [32,64,128,256,512]
            # Integer parameters
            "num_transformer_blocks": ParamRange(1, 4, 1, ParamType.INTEGER),
            "mlp_units": [
                ParamRange(64, 192, 32, ParamType.INTEGER),  # [32,64,128,256,512]
                ParamRange(32, 96, 16, ParamType.INTEGER),  # [16,32,64,128,256]
            ],
            # Continuous parameters
            "dropout": ParamRange(0.1, 0.5, 0.1, ParamType.CONTINUOUS),
            "mlp_dropout": ParamRange(0.1, 0.5, 0.1, ParamType.CONTINUOUS),
            # Loss function parameters
            "focal_gamma": ParamRange(0.5, 4.0, 0.5, ParamType.CONTINUOUS),
            "focal_fp_weight": ParamRange(1.0, 5.0, 0.5, ParamType.CONTINUOUS),
            # Fixed parameters
            "standard_normalizer": ParamRange(False, False, None, ParamType.BOOLEAN),
        }

        # Add validation for param ranges
        self._validate_param_ranges()

    def _get_valid_mlp_units(self, range_def, max_attempts: int = 100) -> List[int]:
        """Get valid MLP unit values where first layer >= second layer.

        Args:
            range_def: List of ParamRange objects for MLP units
            max_attempts: Maximum number of attempts to find valid values

        Returns:
            List of valid MLP unit values

        Raises:
            ValueError: If unable to find valid values after max attempts
        """
        for _ in range(max_attempts):
            values = [r.get_random_value() for r in range_def]
            if values[0] >= values[1]:  # First layer should be >= second layer
                return values
        raise ValueError(
            f"Failed to generate valid MLP units after {max_attempts} attempts. "
            "First layer must be >= second layer."
        )

    def random_params(self) -> Dict[str, Any]:
        """Generate random parameters within defined ranges."""
        params = {}
        for name, range_def in self.param_ranges.items():
            if name == "mlp_units":
                params[name] = self._get_valid_mlp_units(range_def)
            else:
                params[name] = range_def.get_random_value()
        return params

    def _crossover_mlp_units(
        self, parent1_units: List[int], parent2_units: List[int], max_attempts: int = 100
    ) -> List[int]:
        """Perform crossover for MLP units ensuring valid layer sizes.

        Args:
            parent1_units: MLP units from first parent
            parent2_units: MLP units from second parent
            max_attempts: Maximum number of attempts to find valid values

        Returns:
            List of valid MLP unit values from crossover

        Raises:
            ValueError: If unable to find valid values after max attempts
        """
        for _ in range(max_attempts):
            values = [
                p1 if np.random.random() < 0.5 else p2
                for p1, p2 in zip(parent1_units, parent2_units)
            ]
            if values[0] >= values[1]:
                return values
        raise ValueError(
            f"Failed to generate valid MLP units through crossover after {max_attempts} attempts. "
            "First layer must be >= second layer."
        )

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform crossover between two parent parameter sets."""
        child = {}
        for name, param_range in self.param_ranges.items():
            # For fixed ranges (min_val == max_val), use the fixed value
            if (
                not isinstance(param_range, list)
                and param_range.min_val == param_range.max_val
            ):
                child[name] = param_range.min_val
                continue

            if name == "mlp_units":
                child[name] = self._crossover_mlp_units(parent1[name], parent2[name])
            else:
                # For other parameters, randomly choose from either parent
                child[name] = (
                    parent1[name] if np.random.random() < 0.5 else parent2[name]
                )

        return child

    def _mutate_mlp_units(
        self,
        current_units: List[int],
        param_range: List,
        mutation_rate: float,
        max_attempts: int = 100
    ) -> List[int]:
        """Mutate MLP units ensuring valid layer sizes.

        Args:
            current_units: Current MLP unit values
            param_range: List of ParamRange objects for MLP units
            mutation_rate: Probability of mutation for each value
            max_attempts: Maximum number of attempts to find valid values

        Returns:
            List of valid MLP unit values after mutation

        Raises:
            ValueError: If unable to find valid values after max attempts
        """
        for _ in range(max_attempts):
            values = [
                (
                    r.get_random_value()
                    if np.random.random() < mutation_rate
                    else val
                )
                for r, val in zip(param_range, current_units)
            ]
            if values[0] >= values[1]:
                return values
        raise ValueError(
            f"Failed to generate valid MLP units through mutation after {max_attempts} attempts. "
            "First layer must be >= second layer."
        )

    def mutate(self, params: Dict, mutation_rate: float = 0.2) -> Dict:
        """Apply random mutations to parameters based on mutation rate."""
        mutated = params.copy()
        for name, param_range in self.param_ranges.items():
            # Skip mutation for fixed ranges (min_val == max_val)
            if (
                not isinstance(param_range, list)
                and param_range.min_val == param_range.max_val
            ):
                continue

            if np.random.random() < mutation_rate:
                if name == "mlp_units":
                    mutated[name] = self._mutate_mlp_units(
                        params[name], param_range, mutation_rate
                    )
                else:
                    mutated[name] = param_range.get_random_value()

        return mutated

    def _validate_param_ranges(self):
        """Validate parameter ranges are logically consistent"""
        for name, param_range in self.param_ranges.items():
            if isinstance(param_range, list):
                # Handle list of ParamRange objects (e.g. mlp_units)
                for i, range_item in enumerate(param_range):
                    range_item.validate(f"{name}[{i}]")
            else:
                # Handle single ParamRange object
                param_range.validate(name)

    def get_total_combinations(self) -> int:
        """Get total number of possible parameter combinations"""
        total = 1
        for name, param_range in self.param_ranges.items():
            if name == "mlp_units":
                for r in param_range:
                    total *= r.get_combinations()
            else:
                total *= param_range.get_combinations()
        return total

    def set_standard_normalizer(self, value: bool):
        """Set standard_normalizer value."""
        self.param_ranges["standard_normalizer"] = ParamRange(value, value, None, ParamType.BOOLEAN)
