"""
Performance analysis and metrics calculation functions.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Any, Optional
import logging

import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy.interpolate import interp1d

from gpu_hyperparam_scheduler.types import EvaluationMetrics, OptimalThresholdResult, EvaluationMode
from gpu_hyperparam_scheduler.utils import get_class_label, process_threshold
from gpu_hyperparam_scheduler.config import ResultValues
from gpu_hyperparam_scheduler.visualization import print_evaluation_report

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PRCurveResult:
    """Result of precision-recall curve computation."""
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


@dataclass
class FilteredThresholdsResult:
    """Result of threshold filtering."""
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    valid_indices: List[int]


@dataclass
class TestModeResult:
    """Result of test mode processing."""
    optimal_idx: int
    optimal_threshold: float
    optimal_precision: float
    optimal_tp: int


@dataclass
class ValidThresholdData:
    """Container for threshold validation data."""
    threshold: float
    precision: float
    recall: float
    tp: int
    original_index: int  # Maps back to global index


@dataclass
class ModelEvalParams:
    """Parameters for model evaluation."""
    test_gen: Any
    model: Any
    min_threshold: float
    threshold_step: float
    min_tp_ratio: float
    evaluation_mode: EvaluationMode = EvaluationMode.EVALUATE
    verbose: int = 1


@dataclass
class ModelEvaluationResult:
    """Results from model evaluation."""
    metrics: ResultValues
    probabilities: List[np.ndarray]
    true_labels: List[np.ndarray]
    threshold_results: OptimalThresholdResult

def get_distribution_info(data, title="Distribution Information"):
    """
    Prepares distribution information text for a given array of values.

    Parameters:
    data (array-like): An array of numerical values.
    title (str): Title for the distribution section.

    Returns:
    str: Formatted distribution information text.
    """
    # Convert input to a NumPy array for easier calculations
    data = np.array(data)

    # Calculate distribution info
    count = len(data)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    percentiles = np.percentile(data, [10, 25, 50, 75, 90])

    # Format distribution info as a dashboard
    return (
        "----------------------------------------------------------\n"
        f"                      {title}                   \n"
        "----------------------------------------------------------\n"
        f"Count:  {count:<20} | Std Dev: {std_dev:>18.2f}\n"
        f"Mean:   {mean:<20.2f} | Min:     {min_val:>18.2f}\n"
        f"Median: {median:<20.2f} | Max:     {max_val:>18.2f}\n"
        "----------------------------------------------------------\n"
        "                      Percentiles                           \n"
        "----------------------------------------------------------\n"
        f"10th: {percentiles[0]:>10.2f}\n"
        f"25th: {percentiles[1]:>10.2f}\n"
        f"50th: {percentiles[2]:>10.2f} (Median)\n"
        f"75th: {percentiles[3]:>10.2f}\n"
        f"90th: {percentiles[4]:>10.2f}"
    )

def validate_input_arrays(true_labels: np.ndarray, predicted_probabilities: np.ndarray) -> None:
    """
    Validate input arrays for NaN values.

    Args:
        true_labels: Array of true labels
        predicted_probabilities: Array of predicted probabilities

    Raises:
        ValueError: If NaN values are found in either array
    """
    if np.any(np.isnan(predicted_probabilities)):
        nan_indices = np.where(np.isnan(predicted_probabilities))[0]
        logger.error(
            "NaN values detected in predicted probabilities at indices: %s",
            nan_indices[:10] if len(nan_indices) > 10 else nan_indices
        )
        raise ValueError("Input contains NaN values in predicted probabilities")

    if np.any(np.isnan(true_labels)):
        nan_indices = np.where(np.isnan(true_labels))[0]
        logger.error(
            "NaN values detected in true labels at indices: %s",
            nan_indices[:10] if len(nan_indices) > 10 else nan_indices
        )
        raise ValueError("Input contains NaN values in true labels")

def compute_pr_curve(true_labels: np.ndarray, predicted_probabilities: np.ndarray) -> PRCurveResult:
    """
    Compute precision-recall curve and sort values by threshold.

    Args:
        true_labels: Array of true labels
        predicted_probabilities: Array of predicted probabilities

    Returns:
        PRCurveResult containing sorted precision, recall, and thresholds arrays
    """
    logger.info("Computing precision-recall curve")
    precision, recall, thresholds = precision_recall_curve(
        true_labels, predicted_probabilities
    )

    # Sort from highest to lowest threshold
    sorted_indices = np.argsort(thresholds)[::-1]
    return PRCurveResult(
        precision=precision[sorted_indices],
        recall=recall[sorted_indices],
        thresholds=thresholds[sorted_indices]
    )

def filter_thresholds(
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    min_threshold: float,
    threshold_step: float
) -> FilteredThresholdsResult:
    """
    Filter thresholds based on minimum value and step size.

    Args:
        thresholds: Array of thresholds
        precision: Array of precision values
        recall: Array of recall values
        min_threshold: Minimum threshold to consider
        threshold_step: Step size between thresholds

    Returns:
        FilteredThresholdsResult containing filtered arrays and valid indices
    """
    valid_indices = []
    last_kept_threshold = 1

    for i, threshold in enumerate(thresholds):
        if threshold < min_threshold:
            continue
        if abs(last_kept_threshold - threshold) < threshold_step:
            continue
        valid_indices.append(i)
        last_kept_threshold = threshold

    return FilteredThresholdsResult(
        thresholds=thresholds[valid_indices],
        precision=precision[valid_indices],
        recall=recall[valid_indices],
        valid_indices=valid_indices
    )

def calculate_tp_ratio(
    thresholds: np.ndarray,
    filtered_thresholds: np.ndarray,
    filtered_tp_ratio: np.ndarray,
    min_threshold: float
) -> np.ndarray:
    """
    Calculate interpolated true positive ratio for all thresholds.

    Args:
        thresholds: Original thresholds array
        filtered_thresholds: Filtered thresholds array
        filtered_tp_ratio: TP ratio for filtered thresholds
        min_threshold: Minimum threshold to consider

    Returns:
        Array of interpolated TP ratios
    """
    tp_ratio = np.zeros_like(thresholds)
    valid_mask = thresholds >= min_threshold

    if len(filtered_thresholds) > 1:
        f = interp1d(filtered_thresholds, filtered_tp_ratio,
                    kind='linear', fill_value='extrapolate')
        tp_ratio[valid_mask] = f(thresholds[valid_mask])
    elif len(filtered_thresholds) == 1:
        tp_ratio[valid_mask] = filtered_tp_ratio[0]

    return tp_ratio

def process_test_mode(
    predicted_probabilities: np.ndarray,
    true_labels: np.ndarray,
    min_threshold: float,
    thresholds: np.ndarray,
    precision: np.ndarray
) -> TestModeResult:
    """
    Process threshold selection in test mode.

    Args:
        predicted_probabilities: Array of predicted probabilities
        true_labels: Array of true labels
        min_threshold: Minimum threshold to consider
        thresholds: Array of thresholds
        precision: Array of precision values

    Returns:
        TestModeResult containing optimal index, threshold, precision, and true positive count
    """
    optimal_threshold = min_threshold
    predictions = predicted_probabilities >= optimal_threshold
    optimal_tp = np.sum((predictions == 1) & (true_labels == 1))
    optimal_idx = np.abs(thresholds - min_threshold).argmin()
    optimal_precision = precision[optimal_idx]

    return TestModeResult(
        optimal_idx=optimal_idx,
        optimal_threshold=optimal_threshold,
        optimal_precision=optimal_precision,
        optimal_tp=optimal_tp
    )

def get_optimal_threshold_and_pr_curve(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    min_threshold: float,
    threshold_step: float,
    min_tp_ratio: float,
    evaluation_mode: EvaluationMode = EvaluationMode.EVALUATE,
) -> OptimalThresholdResult:
    """
    Calculate the optimal threshold and Precision-Recall Curve.

    Args:
        true_labels: True labels (1D array)
        predicted_probabilities: Predicted probabilities (1D array)
        min_threshold: Minimum threshold to consider
        threshold_step: Step size between thresholds
        min_tp_ratio: Minimum true positive ratio required
        evaluation_mode: Mode of evaluation (EVALUATE or TEST)

    Returns:
        OptimalThresholdResult containing optimal threshold information and curve data
    """
    # Validate input
    validate_input_arrays(true_labels, predicted_probabilities)

    # Compute PR curve
    pr_curve = compute_pr_curve(true_labels, predicted_probabilities)

    # Filter thresholds
    filtered_results = filter_thresholds(
        pr_curve.thresholds, pr_curve.precision, pr_curve.recall, min_threshold, threshold_step
    )

    # Calculate TP ratio for filtered thresholds
    class_one_total = np.sum(true_labels == 1)
    filtered_tp_ratio = np.zeros_like(filtered_results.thresholds)
    for i, threshold in enumerate(filtered_results.thresholds):
        predictions = predicted_probabilities >= threshold
        tp = np.sum((predictions == 1) & (true_labels == 1))
        filtered_tp_ratio[i] = tp / class_one_total if class_one_total else 0.0

    # Calculate full TP ratio array
    tp_ratio = calculate_tp_ratio(
        pr_curve.thresholds, filtered_results.thresholds, filtered_tp_ratio, min_threshold
    )

    min_tp = min_tp_ratio * class_one_total

    # Process based on evaluation mode
    if evaluation_mode == EvaluationMode.TEST:
        test_result = process_test_mode(
            predicted_probabilities, true_labels, min_threshold, pr_curve.thresholds, pr_curve.precision
        )
        optimal_idx = test_result.optimal_idx
        optimal_threshold = test_result.optimal_threshold
        optimal_precision = test_result.optimal_precision
        optimal_tp = test_result.optimal_tp
    else:
        # Process thresholds with ProcessPoolExecutor
        valid_data: List[ValidThresholdData] = []
        logger.info("Processing %d filtered thresholds", len(filtered_results.thresholds))

        with ProcessPoolExecutor(max_workers=max(1, min(4, len(filtered_results.thresholds)))) as executor:
            futures = [
                    executor.submit(
                        process_threshold,
                        threshold,
                        predicted_probabilities,
                        true_labels,
                        filtered_results.precision[i],
                        filtered_results.recall[i],
                        min_tp,
                        i,
                    )
                    for i, threshold in enumerate(filtered_results.thresholds)
            ]

            for future in futures:
                result = future.result()
                if result is None:
                    continue

                valid_data.append(
                    ValidThresholdData(
                        threshold=result.threshold,
                        precision=result.precision,
                        recall=result.recall,
                        tp=result.true_positives,
                        original_index=filtered_results.valid_indices[result.idx]
                    )
                )

        # Find optimal threshold
        if valid_data:
            local_optimal_idx = np.argmax([d.precision for d in valid_data])
            optimal_data = valid_data[local_optimal_idx]
            optimal_threshold = optimal_data.threshold
            optimal_precision = optimal_data.precision
            optimal_idx = optimal_data.original_index
            optimal_tp = optimal_data.tp
        else:
            optimal_idx = np.argmax(pr_curve.precision)
            optimal_threshold = pr_curve.thresholds[optimal_idx]
            optimal_precision = pr_curve.precision[optimal_idx]
            predictions = predicted_probabilities >= optimal_threshold
            optimal_tp = np.sum((predictions == 1) & (true_labels == 1))

    # Log distribution info if needed
    if optimal_tp > min_tp * 2:
        stats_text = (
            "Array Lengths:\n"
            f"{len(true_labels)=} {len(predicted_probabilities)=} {len(pr_curve.thresholds)=}\n"
            f"{get_distribution_info(predicted_probabilities, 'Predicted Probabilities Distribution')}\n"
            f"{get_distribution_info(pr_curve.thresholds, 'Thresholds Distribution')}\n"
            "----------------------------------------------------------"
        )
        logger.warning(stats_text)

    return OptimalThresholdResult(
        optimal_idx=optimal_idx,
        optimal_threshold=optimal_threshold,
        optimal_precision=optimal_precision,
        precision=pr_curve.precision,
        recall=pr_curve.recall,
        thresholds=pr_curve.thresholds,
        tp_ratio=tp_ratio
    )


def evaluate_model_performance(
    params: ModelEvalParams
) -> ModelEvaluationResult:
    """
    Evaluate model performance and return metrics and predictions.

    Args:
        params: ModelEvalParams containing:
            - test_gen: Test data generator
            - model: Trained model
            - min_threshold: Minimum threshold to consider
            - threshold_step: Step size between thresholds
            - min_tp_ratio: Minimum true positive ratio required
            - verbose: Verbosity level for model prediction (default: 1)

    Returns:
        ModelEvaluationResult containing:
        - metrics: ResultValues object with computed metrics
        - probabilities: List of prediction probabilities
        - true_labels: List of true labels
        - threshold_results: OptimalThresholdResult containing optimal threshold information
    """
    # Initialize storage for predictions
    all_probabilities = []
    all_labels = []

    # Collect labels
    logger.info("Collecting labels...")
    all_labels = np.concatenate(
        [params.test_gen.get_labels(i).ravel() for i in range(len(params.test_gen))]
    )
    logger.info("Number of labels: %d", len(all_labels))

    logger.info("Generating predictions...")
    all_probabilities = params.model.predict(params.test_gen, verbose=params.verbose).ravel()
    logger.info("Number of predictions: %d", len(all_probabilities))

    # Check for NaN values in predictions
    if np.any(np.isnan(all_probabilities)):
        nan_indices = np.where(np.isnan(all_probabilities))[0]
        logger.error(
            "NaN values detected in model predictions at indices: %s\n"
            "This usually indicates numerical instability in the model.\n"
            "Common causes:\n"
            "1. Extreme input values causing overflow\n"
            "2. Division by zero in model calculations\n"
            "3. Log of negative numbers\n"
            "4. Gradient explosion during training",
            nan_indices[:10] if len(nan_indices) > 10 else nan_indices
        )
        raise ValueError("Model predictions contain NaN values")

    # Get optimal threshold and PR curve
    logger.info("Finding optimal threshold")
    optimals = get_optimal_threshold_and_pr_curve(
        true_labels=all_labels,
        predicted_probabilities=all_probabilities,
        min_threshold=params.min_threshold,
        threshold_step=params.threshold_step,
        min_tp_ratio=params.min_tp_ratio,
        evaluation_mode=params.evaluation_mode,
    )

    optimal_threshold = optimals.optimal_threshold
    optimal_precision = optimals.optimal_precision

    # Use the optimal threshold to compute class labels
    logger.info("Computing class labels")
    optimal_pred_labels = get_class_label(
        probabilities=all_probabilities,
        confidence_threshold=optimal_threshold,
    )

    # Initialize counters for metrics
    logger.info("Calculating performance metrics")
    total_samples = len(all_labels)
    class_one_total = np.sum(all_labels == 1)
    tp_count = np.sum((optimal_pred_labels == 1) & (all_labels == 1))
    fn_count = np.sum((optimal_pred_labels == 0) & (all_labels == 1))
    fp_count = np.sum((optimal_pred_labels == 1) & (all_labels == 0))
    tn_count = np.sum((optimal_pred_labels == 0) & (all_labels == 0))
    uncertain_count = np.sum(optimal_pred_labels == -1)
    uncertain_class_one = np.sum((optimal_pred_labels == -1) & (all_labels == 1))

    # Calculate final metrics
    certain_predictions = total_samples - uncertain_count

    if certain_predictions > 0:
        accuracy = (tp_count + tn_count) / certain_predictions
    else:
        accuracy = 0.0

    if class_one_total > 0:
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    else:
        recall = 0.0

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

    if np.all(optimals.tp_ratio < params.min_tp_ratio * 0.5):
        precision = 0

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Convert NumPy types to Python native types for JSON serialization
    results = ResultValues(
        total_samples=int(total_samples),
        class_one_total=int(class_one_total),
        certain_predictions=int(certain_predictions),
        uncertain_count=int(uncertain_count),
        true_positives=int(tp_count),
        false_negatives=int(fn_count),
        false_positives=int(fp_count),
        true_negatives=int(tn_count),
        uncertain_class_one=int(uncertain_class_one),
        accuracy=float(accuracy),
        recall=float(recall),
        f1_score=float(f1_score),
        optimal_threshold=float(optimal_threshold),
        optimal_precision=float(optimal_precision),
        precision=float(precision),

        # Calculate ratios with proper denominators
        true_positives_ratio=float(tp_count / class_one_total) if class_one_total > 0 else 0.0,
        false_negatives_ratio=float(fn_count / class_one_total) if class_one_total > 0 else 0.0,
        true_negatives_ratio=float(tn_count / certain_predictions) if certain_predictions > 0 else 0.0,
        false_positives_ratio=float(fp_count / certain_predictions) if certain_predictions > 0 else 0.0,
        class1_ratio=float(class_one_total/total_samples),
        confident_ratio=float(certain_predictions/total_samples),
        uncertain_ratio=float(uncertain_count/total_samples),
    )

    return ModelEvaluationResult(
        metrics=results,
        probabilities=all_probabilities,
        true_labels=all_labels,
        threshold_results=optimals
    )
