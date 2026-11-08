"""
Parameter Optimizer: Handles parameter optimization through gradient descent and binary search.

This module provides functionality for:
- Calculating parameter gradients using finite differences
- Performing binary search optimization in promising regions
- Adapting search strategies based on parameter types and ranges
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from gpu_hyperparam_scheduler.create_model import get_config_id
from gpu_hyperparam_scheduler.types import OptimizationTask, ModelParameters
from gpu_hyperparam_scheduler.hyperparameters import ModelHyperparameters
from gpu_hyperparam_scheduler.param_range import ParamType
from .config import (
    RESULT_TIMEOUT,
    GRADIENT_STEP_RATIO,
)


class ParameterOptimizer:
    """Optimizes parameters using gradient descent and binary search."""

    def __init__(
        self,
        hyperparameters: ModelHyperparameters,
        task_manager,
        min_epochs: int,
    ):
        """Initialize parameter optimizer."""
        self.logger = logging.getLogger(__name__)
        self.hp = hyperparameters
        self.task_manager = task_manager
        self.min_epochs = min_epochs

    def _calculate_step_size(self, param_name: str, current_value: float) -> float:
        """
        Calculate step size for gradient calculation based on parameter type and range.

        Args:
            param_name: Name of parameter
            current_value: Current value of the parameter

        Returns:
            Appropriate step size for the parameter
        """
        param_range = self.hp.param_ranges[param_name]

        if param_range.step is not None:
            return param_range.step
        if param_range.param_type == ParamType.POWER_OF_2:
            return current_value

        return (param_range.max_val - param_range.min_val) * GRADIENT_STEP_RATIO

    def get_gradient_test_points(
        self, param_name: str, current_value: float, num_points: int = 3
    ) -> List[float]:
        """
        Get test points for gradient calculation.

        Args:
            param_name: Name of parameter to calculate gradient for
            current_value: Current value of the parameter

        Returns:
            List of test points to evaluate
        """
        param_range = self.hp.param_ranges[param_name]
        test_points = []

        # Calculate step sizes for the requested number of points
        step_size = self._calculate_step_size(param_name, current_value)
        offsets = np.linspace(-step_size, step_size, num_points)

        if isinstance(param_range, list):
            # For list parameters (like mlp_units), find nearest valid values
            for offset in offsets:
                value = current_value + offset
                value = min(param_range, key=lambda x: abs(x - value))
                if value not in test_points:
                    test_points.append(value)
        else:
            # Handle different parameter types
            if param_range.param_type == ParamType.POWER_OF_2:
                # For power of 2 parameters, find nearest powers
                current_power = int(np.log2(current_value))
                for power in [current_power - 1, current_power, current_power + 1]:
                    value = 2 ** power
                    if param_range.min_val <= value <= param_range.max_val:
                        test_points.append(value)
            elif param_range.param_type == ParamType.INTEGER:
                # For integer parameters, round and clip
                for offset in [-step_size, 0, step_size]:
                    value = int(round(current_value + offset))
                    value = max(param_range.min_val, min(value, param_range.max_val))
                    if value not in test_points:
                        test_points.append(value)
            elif param_range.param_type == ParamType.CONTINUOUS:
                # For continuous parameters, clip to range
                for offset in [-step_size, 0, step_size]:
                    value = current_value + offset
                    value = max(param_range.min_val, min(value, param_range.max_val))
                    if value not in test_points:
                        test_points.append(value)
            # Boolean parameters are not suitable for gradient calculation

        # Remove duplicates while preserving order
        return list(dict.fromkeys(test_points))

    def calculate_gradient_from_results(
        self, param_name: str, results: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate gradient from evaluation results.

        Args:
            param_name: Name of parameter
            results: List of (value, precision) tuples

        Returns:
            Mean gradient
        """
        if len(results) < 2:
            self.logger.warning(
                "Insufficient results to calculate gradient for %s", param_name
            )
            return 0.0

        # Sort results by parameter value
        results.sort(key=lambda x: x[0])

        gradients = []
        for i in range(len(results) - 1):
            v1, p1 = results[i]
            v2, p2 = results[i + 1]
            if v2 - v1 > 0:  # Avoid division by zero
                gradient = (p2 - p1) / (v2 - v1)
                gradients.append(gradient)
                self.logger.debug(
                    "Gradient between %s = %.4f and %.4f: %.4f",
                    param_name, v1, v2, gradient
                )

        mean_gradient = sum(gradients) / len(gradients) if gradients else 0.0
        self.logger.debug(
            "Mean gradient for %s: %.4f (from %d measurements)",
            param_name, mean_gradient, len(results)
        )

        return mean_gradient

    def get_binary_search_points(
        self,
        param_name: str,
        min_val: float,
        max_val: float,
        depth: int = 4
    ) -> List[float]:
        """
        Get test points for binary search.

        Args:
            param_name: Name of parameter to optimize
            min_val: Minimum value of search range
            max_val: Maximum value of search range

        Returns:
            List of test points to evaluate
        """
        param_range = self.hp.param_ranges[param_name]
        mid_val = (min_val + max_val) / 2
        test_points = []

        if isinstance(param_range, list):
            # For list parameters, find nearest valid values
            for p in [min_val, min_val + (max_val - min_val) * 0.25, mid_val,
                     min_val + (max_val - min_val) * 0.75, max_val]:
                value = min(param_range, key=lambda x: abs(x - p))
                if value not in test_points:
                    test_points.append(value)
        else:
            if param_range.param_type == ParamType.POWER_OF_2:
                # For power of 2 parameters, find valid powers within range
                min_power = int(np.ceil(np.log2(min_val)))
                max_power = int(np.floor(np.log2(max_val)))
                powers = np.linspace(min_power, max_power, depth + 1, dtype=int)
                for power in powers:
                    value = 2 ** power
                    if value not in test_points:
                        test_points.append(value)
            elif param_range.param_type in [ParamType.INTEGER, ParamType.CONTINUOUS]:
                # For integer and continuous parameters, use evenly spaced points
                points = np.linspace(min_val, max_val, depth + 1)
                for p in points:
                    if param_range.param_type == ParamType.INTEGER:
                        value = int(round(p))
                    else:
                        value = p
                    value = max(param_range.min_val, min(value, param_range.max_val))
                    if value not in test_points:
                        test_points.append(value)
            # Boolean parameters use direct values (True/False)
            elif param_range.param_type == ParamType.BOOLEAN:
                test_points = [False, True]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(test_points))

    def get_next_binary_search_range(
        self,
        param_name: str,
        min_val: float,
        max_val: float,
        best_value: float,
        results: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Get next binary search range based on results.

        Args:
            param_name: Name of parameter
            min_val: Current minimum value
            max_val: Current maximum value
            best_value: Best value found
            results: List of (value, precision) tuples

        Returns:
            Tuple of (new_min, new_max)
        """
        param_range = self.hp.param_ranges[param_name]
        mid_val = (min_val + max_val) / 2

        if param_range.param_type == ParamType.POWER_OF_2:
            # For power of 2, find adjacent powers
            current_power = int(np.log2(best_value))
            if best_value <= mid_val:
                new_min = max(min_val, 2 ** (current_power - 1))
                new_max = best_value * 2
            else:
                new_min = best_value / 2
                new_max = min(max_val, 2 ** (current_power + 1))
        else:
            # For other types, use standard binary search ranges
            if best_value <= mid_val:
                # Best value in lower half
                new_max = mid_val
                new_min = min_val
                # Include one point from upper half for continuity
                upper_results = [(v, p) for v, p in results if v > mid_val]
                if upper_results:
                    new_max = min(v for v, _ in upper_results)
            else:
                # Best value in upper half
                new_max = max_val
                new_min = mid_val
                # Include one point from lower half for continuity
                lower_results = [(v, p) for v, p in results if v < mid_val]
                if lower_results:
                    new_min = max(v for v, _ in lower_results)

        return new_min, new_max
