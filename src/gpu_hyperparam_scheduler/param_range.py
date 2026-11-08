"""
Parameter range definitions for hyperparameter optimization.
"""

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
import numpy as np


class ParamType(Enum):
    """Parameter type enumeration for range validation and value generation."""
    CONTINUOUS = "continuous"  # Continuous floating point values
    INTEGER = "integer"       # Integer values
    POWER_OF_2 = "power_of_2" # Integer values that must be powers of 2
    BOOLEAN = "boolean"       # Boolean values (True/False)


@dataclass
class ParamRange:
    """
    Range definition for a hyperparameter.

    Supports:
    - Continuous numeric ranges (floating point)
    - Integer ranges
    - Power of 2 ranges
    - Boolean ranges (False -> True)
    - Fixed values (min_val == max_val)
    """

    min_val: Any  # float, int, or bool
    max_val: Any  # Same type as min_val
    step: Optional[float] = None  # Only for numeric ranges
    param_type: ParamType = ParamType.CONTINUOUS

    def get_random_value(self) -> Any:
        """Get random value with improved handling of edge cases"""
        # Handle fixed values
        if self.min_val == self.max_val:
            return self.min_val

        # Handle boolean values
        if self.param_type == ParamType.BOOLEAN:
            return np.random.choice([True, False])

        # Handle power of 2 values
        if self.param_type == ParamType.POWER_OF_2:
            min_pow = int(np.log2(self.min_val))
            max_pow = int(np.log2(self.max_val))
            power = np.random.randint(min_pow, max_pow + 1)
            return 2 ** power

        # Handle numeric values (both continuous and integer)
        if self.step:
            steps = int((self.max_val - self.min_val) / self.step)
            if steps == 0:
                return self.min_val
            possible_values = np.linspace(self.min_val, self.max_val, steps + 1)
            value = np.random.choice(possible_values)
        else:
            value = np.random.uniform(self.min_val, self.max_val)

        # Round to integer if needed
        if self.param_type == ParamType.INTEGER:
            value = int(round(value))
            # Ensure value stays within bounds after rounding
            value = max(self.min_val, min(value, self.max_val))

        return value

    def validate(self, name: str):
        """Validate parameter range is logically consistent"""
        # For fixed values (min_val == max_val), only check type consistency
        if self.min_val == self.max_val:
            return

        # Type-specific validations
        if self.param_type == ParamType.BOOLEAN:
            if not all(isinstance(x, bool) for x in [self.min_val, self.max_val]):
                raise ValueError(f"Boolean parameter {name} must have boolean min/max values")
            if self.min_val and not self.max_val:
                raise ValueError(f"Invalid boolean range for {name}: can only go from False to True")
            return

        # Numeric validations (for CONTINUOUS, INTEGER, and POWER_OF_2)
        if not all(isinstance(x, (int, float)) for x in [self.min_val, self.max_val]):
            raise ValueError(f"Numeric parameter {name} must have numeric min/max values")

        if self.max_val <= self.min_val:
            raise ValueError(f"Invalid range for {name}: max_val must be greater than min_val")

        # Power of 2 validations
        if self.param_type == ParamType.POWER_OF_2:
            if not all(x > 0 for x in [self.min_val, self.max_val]):
                raise ValueError(f"Power of 2 parameters must be positive: {name}")
            if not all(self._is_power_of_2(int(x)) for x in [self.min_val, self.max_val]):
                raise ValueError(f"Min and max values must be powers of 2 for {name}")

        # Step validations for numeric types
        if self.step is not None:
            if self.step <= 0:
                raise ValueError(f"Invalid step for {name}: step must be positive")
            if self.step > (self.max_val - self.min_val):
                raise ValueError(f"Invalid step for {name}: step too large for range")

    def get_combinations(self) -> int:
        """Get number of possible combinations for this parameter range"""
        # For fixed values (min_val == max_val), only one possibility
        if self.min_val == self.max_val:
            return 1

        # For boolean parameters, two possibilities (True/False)
        if self.param_type == ParamType.BOOLEAN:
            return 2

        # For power of 2 parameters
        if self.param_type == ParamType.POWER_OF_2:
            min_pow = int(np.log2(self.min_val))
            max_pow = int(np.log2(self.max_val))
            return max_pow - min_pow + 1

        # For numeric parameters with step
        if self.step:
            return int((self.max_val - self.min_val) / self.step) + 1

        # For continuous ranges, use a reasonable discretization
        return 100  # arbitrary discretization

    def _is_power_of_2(self, n: int) -> bool:
        """Check if a number is a power of 2"""
        if n <= 0:
            return False
        return (n & (n - 1)) == 0
