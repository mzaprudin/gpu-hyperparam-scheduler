import logging
from typing import Any, Dict

import numpy as np

from gpu_hyperparam_scheduler.param_range import ParamRange, ParamType


class ParameterValidator:
    """Handles parameter validation logic."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def is_power_of_2(self, n: int) -> bool:
        """Check if a number is a power of 2."""
        if not isinstance(n, (int, np.integer)) or n <= 0:
            return False
        return (n & (n - 1)) == 0

    def validate_single_value(
        self, param_name: str, value: Any, param_range: ParamRange
    ) -> bool:
        """Validate a single parameter value against its range definition."""
        try:
            if value < param_range.min_val or value > param_range.max_val:
                self.logger.warning(
                    "Parameter %s value %s outside range [%s, %s]",
                    param_name,
                    value,
                    param_range.min_val,
                    param_range.max_val,
                )
                return False

            if param_range.param_type == ParamType.POWER_OF_2:
                if not self.is_power_of_2(value):
                    self.logger.warning(
                        "Parameter %s value %s must be a power of 2", param_name, value
                    )
                    return False
            elif param_range.param_type == ParamType.INTEGER:
                if not isinstance(value, (int, np.integer)):
                    self.logger.warning(
                        "Parameter %s value %s must be an integer", param_name, value
                    )
                    return False
            elif param_range.param_type == ParamType.BOOLEAN:
                if not isinstance(value, bool):
                    self.logger.warning(
                        "Parameter %s value %s must be a boolean", param_name, value
                    )
                    return False
            return True
        except Exception as e:
            self.logger.warning(
                "Error validating %s = %s: %s", param_name, value, str(e)
            )
            return False

    def validate_parameter_values(self, params: Dict, param_ranges: Dict) -> bool:
        """Validate parameter values against defined ranges."""
        try:
            for param_name, value in params.items():
                if param_name not in param_ranges:
                    self.logger.warning(
                        "Unknown parameter in loaded results: %s", param_name
                    )
                    return False

                param_range = param_ranges[param_name]
                if isinstance(param_range, list):
                    if not isinstance(value, list) or len(value) != len(param_range):
                        self.logger.warning(
                            "Invalid value for parameter %s: expected list of length %d, got %s",
                            param_name,
                            len(param_range),
                            value,
                        )
                        return False

                    for val, range_obj in zip(value, param_range):
                        if not self.validate_single_value(param_name, val, range_obj):
                            return False
                else:
                    if not self.validate_single_value(param_name, value, param_range):
                        return False
            return True
        except Exception as e:
            self.logger.warning("Error validating parameter values: %s", str(e))
            return False
