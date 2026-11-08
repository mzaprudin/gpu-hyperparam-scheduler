"""
Configuration classes for model training and evaluation.
"""

import logging
from dataclasses import dataclass, fields
from typing import List, Optional
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class Columns(Enum):
    ACCOUNT = "account"

class Features(Enum):
    BLOCK_NUMBER = "block_number"
    BLOCK_TIME = "block_time"
    TOKEN = "token"
    TIMESTAMP = "unix_timestamp"
    TRANSACTIONS = "transactions"
    ACCOUNTS = "accounts"
    BUY_VOLUME = "buy_volume"
    SELL_VOLUME = "sell_volume"
    PRICE = "price"
    LIQUIDITY_TOKEN = "liquidity_token"
    LIQUIDITY_SOL = "liquidity_sol"
    Y_UP = "y_up"
    Y_DN = "y_dn"
    Y_CLASS = "y_class"
    FILTER = "filter"
    Y_LIQ = "y_liq"

@dataclass
class TrainingParameters:
    """
    Configuration parameters for model training.

    Attributes:
        gpu_idx: Index of GPU to use
        name: Name for saving model files
        data_file: Path to input data file
        save_dir: Directory for saving model outputs
        test_hours: Number of hours for test set
        train_hours: Number of hours for training set
        hours_offset: Hours offset from latest timestamp in history table
        val_ratio: Validation set ratio
        model_to_load: Path to pre-trained model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        sequence_length: Length of input sequences
        step: Step size between sequences
        up_more_than: Threshold for upward movement
        down_less_than: Threshold for downward movement
        min_price_range_percent: Minimum price range filter
        min_trading_volume: Minimum trading volume filter
        reverse_volume_filter: Reverse volume filter flag
        min_tp_ratio: Minimum true positive ratio
        min_threshold: Minimum confidence threshold
        threshold_step: Step size for threshold search
        test_only: Run only testing flag
        standard_normalizer: Use standard normalization
        features: List of feature column names
        label: Label column name
        limit: Tokens to load
    """

    gpu_idx: int = 0
    name: str = None
    data_file: str = "data/demo.pkl"
    save_dir: str = "artifacts"
    train_hours: int = 336  # Default 14 days of training data
    val_hours: int = 96
    test_hours: int = 120
    hours_offset: int = 0  # Hours offset from latest timestamp
    model_to_load: str = None
    batch_size: int = 1024
    num_epochs: int = 10
    sequence_length: int = 200
    forward_length: int = 75
    skip_length: int = 15
    step: int = 1
    up_more_than: float = 0.10
    down_less_than: float = 0.05
    min_price_range_percent: float = 0.1
    min_liquidity: float = 100
    life_hours: int = 24
    min_trading_volume: float = 400
    reverse_volume_filter: bool = False
    min_tp_ratio: float = 0.05
    min_threshold: float = 0.65
    threshold_step: float = 0.01
    test_only: bool = False
    standard_normalizer: bool = False
    features: List[str] = None
    label: str = Features.Y_CLASS.value
    filter_column: str = Features.FILTER.value
    limit: int = 0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [Features.TIMESTAMP.value, Features.TRANSACTIONS.value, Features.ACCOUNTS.value,
                             Features.BUY_VOLUME.value, Features.SELL_VOLUME.value, Features.PRICE.value,
                             Features.LIQUIDITY_TOKEN.value, Features.LIQUIDITY_SOL.value]


@dataclass
class ResultValues:
    """
    Container for model evaluation metrics.

    Attributes:
        total_samples: Total number of samples evaluated
        class_one_total: Total number of positive class samples
        certain_predictions: Number of confident predictions
        uncertain_count: Number of uncertain predictions
        true_positives: Number of true positive predictions
        false_negatives: Number of false negative predictions
        false_positives: Number of false positive predictions
        true_negatives: Number of true negative predictions
        uncertain_class_one: Number of uncertain positive class predictions
        accuracy: Model accuracy (excluding uncertain predictions)
        recall: Model recall
        f1_score: F1 score
        optimal_threshold: Optimal confidence threshold
        optimal_precision: Precision at optimal threshold
        precision: Model precision
    """

    total_samples: int
    class_one_total: int
    certain_predictions: int
    uncertain_count: int
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    uncertain_class_one: int
    accuracy: float
    recall: float
    f1_score: float
    optimal_threshold: float
    optimal_precision: float
    precision: float

    true_positives_ratio: float
    false_negatives_ratio: float
    true_negatives_ratio: float
    false_positives_ratio: float
    class1_ratio: float
    confident_ratio: float
    uncertain_ratio: float


def dataclass_to_string(data, float_precision: int = 3) -> str:
    """
    Converts a dataclass instance to a string representation.

    Args:
        data: An instance of a dataclass.
        float_precision: Number of decimal places to format floating-point numbers.

    Returns:
        A string representation of the dataclass fields and their values.
    """
    result = []
    for field in fields(data):
        value = getattr(data, field.name)
        if isinstance(value, float):
            # Format floating-point numbers
            value = f"{value:.{float_precision}f}"
        result.append(f"{field.name}: {value}")
    return "\n".join(result)


def print_dataclass(data) -> None:
    """Print a formatted representation of a dataclass instance."""
    logger.info("\n%s", dataclass_to_string(data))
