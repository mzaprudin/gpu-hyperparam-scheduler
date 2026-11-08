"""
Knowledge Nexus: Manages and analyzes historical optimization results.

This module maintains a knowledge base of optimization results, including:
- Parameter gradients and sensitivity information
- Performance regions for parameter configurations
- Best configurations found (both loaded and new)
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from gpu_hyperparam_scheduler.types import OptimizationResult

@dataclass
class SortedParameter:
    """Represents a parameter sorted by sensitivity and exploration time."""
    name: str
    sensitivity: float
    last_explored_time: float

class ParameterGradient:
    """Tracks parameter gradients and sensitivity information."""

    def __init__(self, param_name: str):
        self.param_name: str = param_name
        self.gradients: List[Tuple[float, float]] = []  # List of (value, gradient) pairs
        self.sensitivity: float = 0.0  # Measure of parameter's impact on performance

    def add_gradient(self, value: float, gradient: float) -> None:
        """Add a new gradient measurement."""
        self.gradients.append((value, gradient))

    def update_sensitivity(self) -> None:
        """Update parameter sensitivity based on gradient magnitudes."""
        if self.gradients:
            self.sensitivity = np.mean([abs(g) for _, g in self.gradients])


class KnowledgeNexus:
    """Manages and analyzes historical optimization results."""

    def __init__(self):
        self.param_gradients: Dict[str, ParameterGradient] = {}  # Parameter name -> ParameterGradient
        self.performance_regions: Dict[float, List[OptimizationResult]] = defaultdict(list)  # Performance -> List of configs
        self.results: List[OptimizationResult] = []  # List of results
        self.loaded_count: int = 0  # Number of loaded results
        self.new_count: int = 0  # Number of new results
        self.new_configs: Set[str] = set()  # Set of config_ids for new results

    def process_gradient_result(
        self, result: OptimizationResult, param_name: str, gradient: float
    ) -> None:
        """
        Process gradient result and update parameter sensitivity.

        Args:
            result: The optimization result to process
            param_name: Name of the parameter being processed
            gradient: Gradient value for the parameter

        Raises:
            KeyError: If param_name is not found in result.params
        """
        if param_name not in self.param_gradients:
            self.param_gradients[param_name] = ParameterGradient(param_name)

        params_dict: Dict[str, Any] = asdict(result.params)
        if param_name not in params_dict:
            raise KeyError(f"Parameter '{param_name}' not found in optimization result parameters")

        self.param_gradients[param_name].add_gradient(params_dict[param_name], gradient)
        self.param_gradients[param_name].update_sensitivity()

    def add_result(self, result: OptimizationResult, is_loaded: bool = False) -> None:
        """Add new result to knowledge base."""
        # Track high-performing regions
        perf_bucket: float = round(result.precision, 2)
        self.performance_regions[perf_bucket].append(result)

        # Update counters
        if is_loaded:
            self.loaded_count += 1
        else:
            self.new_count += 1

        # Update best configurations with preference for new results
        if not is_loaded:
            self.new_configs.add(result.config_id)
        self.results.append(result)
        # Sort by effective precision but keep original precision in records
        self.results.sort(key=self._effective_precision, reverse=True)

    def find_explored_parameter(
        self, result_params: Dict[str, Any], base_params: Dict[str, Any]
    ) -> Tuple[Optional[str], List[Tuple[float, float]]]:
        """
        Find which parameter was being explored and gather its recent results.

        Args:
            result_params: Parameters from the current result
            base_params: Base parameters used for comparison

        Returns:
            Tuple of (explored parameter name or None, list of (value, precision) pairs)
        """
        # Find the parameter that differs from base params
        for param_name in result_params:
            if param_name not in base_params:
                continue

            if result_params[param_name] == base_params[param_name]:
                continue

            # Get all recent results for this parameter
            recent_results: List[Tuple[float, float]] = []
            for result in self.results[:20]:  # Use last 20 results
                params_dict = asdict(result.params)
                if param_name in params_dict:
                    recent_results.append((params_dict[param_name], result.precision))

            return param_name, recent_results

        return None, []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "loaded_results": self.loaded_count,
            "new_results": self.new_count,
            "total_results": self.loaded_count + self.new_count,
            "best_precision": self.results[0].precision if self.results else 0.0,
            "num_regions": len(self.performance_regions),
        }

    def get_promising_regions(self, percentile: float = 0.2) -> List[OptimizationResult]:
        """
        Identify promising parameter regions based on historical results.

        Args:
            percentile: Fraction of top-performing regions to consider (0.0 to 1.0)

        Returns:
            List of OptimizationResult objects from promising regions

        Raises:
            ValueError: If percentile is not between 0 and 1
        """
        if not 0 <= percentile <= 1:
            raise ValueError("Percentile must be between 0 and 1")

        if not self.performance_regions:
            return []

        # Get configurations from top percentile of performance buckets
        perf_thresholds: List[float] = sorted(self.performance_regions.keys(), reverse=True)
        threshold_idx = max(1, int(len(perf_thresholds) * percentile))
        top_threshold: float = perf_thresholds[threshold_idx - 1]

        promising_results: List[OptimizationResult] = []
        for perf, configs in self.performance_regions.items():
            if perf >= top_threshold:
                promising_results.extend(configs)

        return promising_results

    def update_parameter_sensitivities(self) -> None:
        """Update sensitivity values for all parameters."""
        for gradient in self.param_gradients.values():
            gradient.update_sensitivity()

    def set_parameter_sensitivity(self, param_name: str, sensitivity: float) -> None:
        """Set sensitivity value for a parameter."""
        if param_name not in self.param_gradients:
            self.param_gradients[param_name] = ParameterGradient(param_name)
        self.param_gradients[param_name].sensitivity = sensitivity

    def get_parameter_sensitivities(self) -> Dict[str, float]:
        """Get dictionary of parameter sensitivities."""
        return {name: grad.sensitivity for name, grad in self.param_gradients.items()}

    def get_best_results(self, limit: Optional[int] = None) -> List[OptimizationResult]:
        """
        Get best results sorted by precision.

        Args:
            limit: Optional maximum number of results to return

        Returns:
            List of OptimizationResult objects sorted by precision (high to low)
        """
        if limit:
            return self.results[:limit]
        return self.results

    def get_result_count(self, new_configs_only: bool = False) -> int:
        """
        Get total number of results.

        Args:
            new_configs_only: If True, only count results from new configurations

        Returns:
            Number of results matching the criteria
        """
        if new_configs_only:
            return sum(1 for result in self.results if result.config_id in self.new_configs)
        return len(self.results)

    def get_sorted_parameters(self, last_explored: Dict[str, float]) -> List[SortedParameter]:
        """
        Get parameters sorted by sensitivity and last exploration time.

        Args:
            last_explored: Dictionary mapping parameter names to their last exploration time

        Returns:
            List of SortedParameter objects sorted by sensitivity (high to low)
            and exploration time (old to new)
        """
        sorted_params = [
            SortedParameter(
                name=name,
                sensitivity=grad.sensitivity,
                last_explored_time=last_explored.get(name, 0)
            )
            for name, grad in self.param_gradients.items()
        ]
        sorted_params.sort(key=lambda x: (x.sensitivity, -x.last_explored_time), reverse=True)  # Sort by sensitivity (high to low) and time (old to new)
        return sorted_params

    def _effective_precision(self, result: OptimizationResult, new_config_bonus: float = 0.01) -> float:
        """
        Calculate effective precision with configurable bonus for new configurations.

        Args:
            result: The optimization result to calculate precision for
            new_config_bonus: Bonus precision added for new configurations

        Returns:
            Effective precision value
        """
        return result.precision + new_config_bonus if result.config_id in self.new_configs else result.precision
