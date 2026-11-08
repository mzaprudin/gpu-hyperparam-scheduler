"""
Utility functions for model training and evaluation.
"""

import os
import random
import logging
import time
from typing import Optional
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from pynvml import NVMLError, nvmlInit, nvmlDeviceGetCount, nvmlShutdown

logger = logging.getLogger(__name__)

def get_gpus_amount() -> int:
    try:
        nvmlInit()
    except NVMLError:
        return 0
    try:
        return nvmlDeviceGetCount()
    finally:
        nvmlShutdown()


def get_class_label(probabilities, confidence_threshold) -> int:
    """
    Convert probability values into class labels based on a confidence threshold.

    Args:
        probabilities (float, list, or np.ndarray): Probability values (single number, list, or array).
        confidence_threshold (float): Threshold for assigning class labels.

    Returns:
        int or np.ndarray: Class labels (1, 0, or -1).
    """
    # Convert input to a NumPy array for consistent processing
    probabilities = np.asarray(probabilities)

    # Assign class labels based on the confidence threshold
    class_labels = np.where(
        probabilities >= confidence_threshold,  # Condition for class 1
        1,  # Value if condition is True
        np.where(
            probabilities <= 1 - confidence_threshold,  # Condition for class 0
            0,  # Value if condition is True
            -1,  # Value if neither condition is True
        ),
    )

    # Return a single value if the input was a single number
    if np.isscalar(probabilities):
        return int(class_labels)

    return class_labels


@dataclass
class ThresholdResult:
    """Result of threshold processing containing metrics and counts."""
    threshold: float
    precision: float
    recall: float
    idx: int
    true_positives: int

def process_threshold(
    threshold: float,
    predicted_probabilities: np.ndarray,
    true_labels: np.ndarray,
    precision: float,
    recall: float,
    min_tp: int,
    idx: int,
) -> Optional[ThresholdResult]:
    """
    Process a single threshold to calculate metrics.

    Args:
        threshold (float): Threshold value.
        predicted_probabilities (np.ndarray): Predicted probabilities.
        true_labels (np.ndarray): True labels.
        precision (float): Precision value for the threshold.
        recall (float): Recall value for the threshold.
        min_tp (int): Minimum required true positives.
        idx (int): Index for tracking position.

    Returns:
        Optional[ThresholdResult]: Dataclass containing threshold metrics if TP >= min_tp; otherwise, None.
    """
    # Use get_class_label to convert probabilities to class labels
    pred_labels = get_class_label(
        probabilities=predicted_probabilities, confidence_threshold=threshold
    )

    # Calculate true positives
    tp = np.sum((pred_labels == 1) & (true_labels == 1))

    # Check if TP meets the minimum requirement
    if tp >= min_tp:
        return ThresholdResult(
            threshold=threshold,
            precision=precision,
            recall=recall,
            idx=idx,
            true_positives=tp
        )
    return None


def set_seeds(seed: Optional[int] = 42) -> int:
    if seed is None:
        seed = int(time.time())

    # Python's built-in random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For complete reproducibility in TensorFlow operations
    try:
        # TF 2.x
        tf.keras.utils.set_random_seed(seed)
    except:
        pass

    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

    # Optional: for GPU operations (if using)
    try:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except:
        pass

    return seed


class TrainerError(Exception):
    """Custom exception for worker process errors."""

    pass
