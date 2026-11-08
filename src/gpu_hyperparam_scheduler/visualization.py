"""Visualization functions for model analysis and evaluation."""

from dataclasses import dataclass
from typing import Optional
import logging
import matplotlib.pyplot as plt
import numpy as np

from gpu_hyperparam_scheduler.types import OptimalThresholdResult
from gpu_hyperparam_scheduler.config import dataclass_to_string, ResultValues

@dataclass
class PlotParameters:
    """Parameters for probability distribution visualization."""

    probabilities: np.ndarray
    true_labels: np.ndarray
    optimals: OptimalThresholdResult
    params: Optional[object] = None
    model_params: Optional[object] = None
    results: Optional[object] = None
    file_name_pic: Optional[str] = None


# Configure logging
logger = logging.getLogger(__name__)


def analyze_probability_distribution(plot_params: PlotParameters) -> None:
    """
    Analyze probability distribution and create visualization plots.

    Args:
        plot_params: Parameters for visualization including probabilities, labels,
                  thresholds, and optional display configuration
    """
    # Split by class
    prob_class_0 = plot_params.probabilities[plot_params.true_labels == 0]
    prob_class_1 = plot_params.probabilities[plot_params.true_labels == 1]

    # Create visualization
    plt.figure(figsize=(18, 18))

    # Top left plot (Probability Distribution)
    plt.subplot(221)
    plt.hist(prob_class_0, bins=50, alpha=0.5, label="Class 0")
    plt.hist(prob_class_1, bins=50, alpha=0.5, label="Class 1")
    plt.axvline(
        x=plot_params.optimals.optimal_threshold,
        color="r",
        linestyle="--",
        label="Confidence Threshold",
    )
    plt.title("Probability Distribution by Class")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.legend()

    # Top right plot (Threshold-Precision and TP Ratio Curves)
    plt.subplot(222)

    # Plot Precision curve
    plt.plot(
        plot_params.optimals.thresholds,
        plot_params.optimals.precision,
        linestyle="-",
        linewidth=1,
        label="Threshold-Precision Curve",
        color='blue'
    )

    # Plot TP ratio curve
    plt.plot(
        plot_params.optimals.thresholds,
        plot_params.optimals.tp_ratio,
        linestyle="-",
        linewidth=1,
        label="Threshold-TP Ratio Curve",
        color='green'
    )

    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Threshold-Precision and TP Ratio Curves")

    # Highlight the optimal threshold on both curves
    plt.scatter(
        plot_params.optimals.optimal_threshold,
        plot_params.optimals.optimal_precision,
        color="red",
        label=(
            f"Optimal Threshold: {plot_params.optimals.optimal_threshold:.2f}\n"
            f"Optimal Precision: {plot_params.optimals.optimal_precision:.2f}\n"
            f"TP Ratio: {plot_params.optimals.tp_ratio[plot_params.optimals.optimal_idx]:.2f}"
        ),
    )
    plt.scatter(
        plot_params.optimals.optimal_threshold,
        plot_params.optimals.tp_ratio[plot_params.optimals.optimal_idx],
        color="red",
        marker='s'
    )
    plt.legend()

    # Bottom left plot (Hyperparameters)
    plt.subplot(223)
    if plot_params.params:
        plt.axis("off")
        text = ""
        if plot_params.params:
            text += f"Training Parameters:\n{dataclass_to_string(plot_params.params)}\n\n"

        # Create title separately and position text right below it
        plt.title("Hyperparameters", pad=20)
        plt.text(
            0.1,
            0.95,
            text,
            fontsize=12,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            linespacing=1.5,
        )

    # Bottom right plot (Results)
    plt.subplot(224)
    if plot_params.results or plot_params.model_params:
        plt.axis("off")
        text = ""
        if plot_params.results:
            text = dataclass_to_string(plot_params.results)
        if plot_params.model_params:
            text += f"\n\nModel Parameters:\n{dataclass_to_string(plot_params.model_params)}"

        # Create title separately and position text right below it
        plt.title("Results", pad=20)
        plt.text(
            0.1,
            0.95,
            text,
            fontsize=16,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            linespacing=1.5,
        )

    plt.tight_layout()

    if plot_params.file_name_pic:
        plt.savefig(plot_params.file_name_pic)
    plt.close()


def print_evaluation_report(metrics: ResultValues) -> None:
    """
    Print a detailed evaluation report with model performance metrics.

    Args:
        metrics: ResultValues dataclass containing all evaluation metrics
                including sample counts, predictions, and performance scores
    """

    # Print detailed report in a single call
    logger.info(
        "\nModel Evaluation Report:"
        "\n%s"
        "\n%20s %20s"
        "\n%s"
        "\nSample Distribution:"
        "\n%20s %11d"
        "\n%20s %11d %8.4f"
        "\n%s"
        "\nPrediction Analysis:"
        "\n%20s %11d %8.4f"
        "\n%20s %11d %8.4f"
        "\n%s"
        "\nClass 1 Performance:"
        "\n%20s %11d %8.4f"
        "\n%20s %11d %8.4f"
        "\n%20s %11d %8.4f"
        "\n%20s %11d %8.4f"
        "\n%s"
        "\nMetrics:"
        "\n%20s %20.4f"
        "\n%20s %20.4f"
        "\n%20s %20.4f"
        "\n%20s %20.4f"
        "\n%20s %20.4f"
        "\n%20s %20.4f"
        "\n%s",
        "="*50,
        "Metric", "Values",
        "-"*50,
        "Total samples:", metrics.total_samples,
        "Class 1 samples:", metrics.class_one_total, metrics.class1_ratio,
        "-"*50,
        "Confident:", metrics.certain_predictions, metrics.confident_ratio,
        "Uncertain:", metrics.uncertain_count, metrics.uncertain_ratio,
        "-"*50,
        "True Positives:", metrics.true_positives, metrics.true_positives_ratio,
        "False Negatives:", metrics.false_negatives, metrics.false_negatives_ratio,
        "Uncertain:", metrics.uncertain_class_one, metrics.uncertain_ratio,
        "False Positives:", metrics.false_positives, metrics.false_positives_ratio,
        "-"*50,
        "Uncertain Rate:", metrics.uncertain_ratio,
        "F1 Score:", metrics.f1_score,
        "Recall:", metrics.recall,
        "Accuracy:", metrics.accuracy,
        "Threshold:", metrics.optimal_threshold,
        "Precision:", metrics.precision,
        "="*50
    )


def print_comparison_report(evaluation: ResultValues, test: ResultValues) -> None:
    """
    Print a comparison report between evaluation and test metrics.

    Args:
        evaluation: ResultValues dataclass containing evaluation metrics
        test: ResultValues dataclass containing test metrics
    """

    # Print comparison report
    logger.info(
        "\nModel Comparison Report:"
        "\n%s"
        "\n%20s %20s %20s"
        "\n%s"
        "\nSample Distribution:"
        "\n%20s %11d %20d"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%s"
        "\nPrediction Analysis:"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%s"
        "\nClass 1 Performance:"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%20s %11d %8.4f %11d %8.4f"
        "\n%s"
        "\nMetrics:"
        "\n%20s %20.4f %20.4f"
        "\n%20s %20.4f %20.4f"
        "\n%20s %20.4f %20.4f"
        "\n%20s %20.4f %20.4f"
        "\n%20s %20.4f %20.4f"
        "\n%20s %20.4f %20.4f"
        "\n%s",
        "="*80,
        "Metric", "Evaluation", "Test",
        "-"*80,
        "Total samples:", evaluation.total_samples, test.total_samples,
        "Class 1 samples:", evaluation.class_one_total, evaluation.class1_ratio, test.class_one_total, test.class1_ratio,
        "-"*80,
        "Confident:", evaluation.certain_predictions, evaluation.confident_ratio, test.certain_predictions, test.confident_ratio,
        "Uncertain:", evaluation.uncertain_count, evaluation.uncertain_ratio, test.uncertain_count, test.uncertain_ratio,
        "-"*80,
        "True Positives:", evaluation.true_positives, evaluation.true_positives_ratio, test.true_positives, test.true_positives_ratio,
        "False Negatives:", evaluation.false_negatives, evaluation.false_negatives_ratio, test.false_negatives, test.false_negatives_ratio,
        "Uncertain:", evaluation.uncertain_class_one, evaluation.uncertain_ratio, test.uncertain_class_one, test.uncertain_ratio,
        "False Positives:", evaluation.false_positives, evaluation.false_positives_ratio, test.false_positives, test.false_positives_ratio,
        "-"*80,
        "Uncertain Rate:", evaluation.uncertain_ratio, test.uncertain_ratio,
        "F1 Score:", evaluation.f1_score, test.f1_score,
        "Recall:", evaluation.recall, test.recall,
        "Accuracy:", evaluation.accuracy, test.accuracy,
        "Threshold:", evaluation.optimal_threshold, test.optimal_threshold,
        "Precision:", evaluation.precision, test.precision,
        "="*80
    )
