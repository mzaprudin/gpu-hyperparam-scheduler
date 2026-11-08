"""
Data loading and preparation functions for machine learning tasks.

This module provides functionality for:
- Loading data from pickle files
- Preparing train/validation/test splits
- Creating data generators for batch processing
- Managing master generators for time series data

The module handles both raw dataframes and pre-saved generators, with automatic
saving of created generators for future use.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from dataclasses import dataclass

from gpu_hyperparam_scheduler.utils import set_seeds
from gpu_hyperparam_scheduler.data_generator import MasterGenerator, MultiPersonTimeSeriesGenerator
from gpu_hyperparam_scheduler.data_preparation import (
    create_data_generators_from_split,
    calculate_filters,
    recalculate_labels,
    prepare_train_val_test_split,
)
from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.constants import (
    TRAIN_MASTER,
    VAL_MASTER,
    TEST_MASTER,
    PARAMS,
    MASTER_GENERATOR_KEYS,
    MASTER_GENERATOR_SUFFIX,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_data(params: TrainingParameters) -> Dict[str, pd.DataFrame]:
    """
    Load data from pickle file and prepare it for use by applying filters and recalculating labels.

    Args:
        params: TrainingParameters object containing configuration settings including:
            - data_file: Path to the pickle file containing the data
            - limit: Optional limit on number of dataframes to load
            - sequence_length: Window size for calculating filters
            - min_trading_volume: Threshold for trading volume filter
            - min_price_range_percent: Required price range percentage
            - reverse_volume_filter: Whether to reverse volume filter logic
            - up_more_than: Threshold for upward price movement
            - down_less_than: Threshold for downward price movement

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping identifiers to prepared dataframes
            with calculated filters and recalculated labels

    Raises:
        FileNotFoundError: If the data file doesn't exist
        pickle.UnpicklingError: If the pickle file is corrupted or invalid
    """
    logger.info(f"Loading data from {params.data_file}")
    with open(params.data_file, "rb") as file:
        data = pickle.load(file)  # nosec B301 - Used only with trusted data files
    logger.info(f"{len(data)} dataframes were loaded")

    if params.limit is not None and params.limit > 0:
        data = dict(list(data.items())[: params.limit])

    # Calculate filters and recalculate labels once
    logger.info("Creating filter labels")
    calculate_filters(
        dataframes_dict=data,
        window_size=params.sequence_length,
        min_liquidity=params.min_liquidity,
        trading_volume_threshold=params.min_trading_volume,
        range_percent=params.min_price_range_percent,
        reverse=params.reverse_volume_filter,
    )

    logger.info("Recalculating labels")
    recalculate_labels(
        dataframes_dict=data,
        forward_length=params.forward_length,
        skip_length=params.skip_length,
        up_more_than=params.up_more_than,
        down_less_than=params.down_less_than,
        label_column=params.label,
        min_liquidity=params.min_liquidity,
    )

    return data


def load_frames(
    params: TrainingParameters,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load and prepare data, then split into train/validation/test sets.

    This function is a convenience wrapper that:
    1. Loads data using load_data()
    2. Applies necessary preprocessing
    3. Splits the data into training, validation and test sets

    Args:
        params: TrainingParameters object containing configuration settings including:
            - All parameters required by load_data()
            - test_hours: Number of hours to use for test set
            - val_ratio: Ratio of remaining data to use for validation

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]: A tuple containing:
            - train_dfs: List of dataframes for training
            - val_dfs: List of dataframes for validation
            - test_dfs: List of dataframes for testing
            Each dataframe contains preprocessed data with filters and labels

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If test_hours or val_ratio are invalid
    """
    # Load and prepare data
    data = load_data(params)

    # Split data into train/val/test sets
    logger.info("Splitting data into train/val/test sets")
    train_dfs, val_dfs, test_dfs = prepare_train_val_test_split(
        dataframes_dict=data,
        test_hours=params.test_hours,
        val_hours=params.val_hours,
    )

    return train_dfs, val_dfs, test_dfs


def create_master_generators(
    params: TrainingParameters,
    train_dfs: List[pd.DataFrame],
    val_dfs: List[pd.DataFrame],
    test_dfs: List[pd.DataFrame],
) -> Tuple[MasterGenerator, MasterGenerator, MasterGenerator]:
    """
    Create MasterGenerator objects for training, validation, and testing.

    MasterGenerator objects handle the generation of sequences from multiple
    dataframes for time series analysis. They manage window creation, feature
    selection, and label handling.

    Args:
        params: TrainingParameters object containing:
            - features: List of feature column names
            - label: Name of the label column
            - sequence_length: Size of the sliding window
            - step: Step size for sliding window
            - filter_column: Name of the column used for filtering
        train_dfs: List of training dataframes
        val_dfs: List of validation dataframes
        test_dfs: List of test dataframes

    Returns:
        Tuple[MasterGenerator, MasterGenerator, MasterGenerator]: A tuple containing:
            - train_master: MasterGenerator for training data
            - val_master: MasterGenerator for validation data
            - test_master: MasterGenerator for test data
            Each generator is configured with the same parameters but different data

    Raises:
        ValueError: If any of train_dfs, val_dfs, or test_dfs is None
        ValueError: If required parameters in TrainingParameters are missing
    """
    if any(dfs is None for dfs in [train_dfs, val_dfs, test_dfs]):
        raise ValueError("All dataframe lists (train_dfs, val_dfs, test_dfs) must be provided")

    set_seeds(params.seed)

    logger.info("Creating master generators")

    # Create master generators
    train_master = MasterGenerator(
        dataframes_list=train_dfs,
        feature_columns=params.features,
        window_size=params.sequence_length,
        step=params.step,
        label_column=params.label,
        filter_column=params.filter_column,
    )

    val_master = MasterGenerator(
        dataframes_list=val_dfs,
        feature_columns=params.features,
        window_size=params.sequence_length,
        step=params.step,
        label_column=params.label,
        filter_column=params.filter_column,
    )

    test_master = MasterGenerator(
        dataframes_list=test_dfs,
        feature_columns=params.features,
        window_size=params.sequence_length,
        step=params.step,
        label_column=params.label,
        filter_column=params.filter_column,
    )

    return train_master, val_master, test_master


@dataclass
class TrainingMasterGenerators:
    train: MasterGenerator
    val: MasterGenerator
    test: MasterGenerator
    path: str
    params: TrainingParameters

def load_master_generators(params: TrainingParameters) -> TrainingMasterGenerators:
    """
    Load MasterGenerators from a file, handling both pre-saved generators and raw dataframes.

    This function attempts to load pre-saved MasterGenerators first. If not found,
    it creates new generators from raw dataframes and automatically saves them
    for future use.

    Args:
        params: TrainingParameters object containing:
            - data_file: Path to the pickle file
            - All parameters required by get_master_generators()

    Returns:
        TrainingMasterGenerators: A dataclass containing:
            - train: MasterGenerator for training data
            - val: MasterGenerator for validation data
            - test: MasterGenerator for test data
            - path: Path to the saved generators file

    Raises:
        FileNotFoundError: If the data file doesn't exist
        pickle.UnpicklingError: If the pickle file is corrupted
    """
    logger.info(f"Loading data from {params.data_file}")
    with open(params.data_file, "rb") as file:
        data = pickle.load(file)  # nosec B301 - Used only with trusted data files

    return get_master_generators(params=params, data=data)


def get_master_generators(
    params: TrainingParameters,
    data: Dict
) -> TrainingMasterGenerators:
    """
    Process input data and return MasterGenerators, handling both pre-saved generators and raw data.

    This function:
    1. Checks if input contains pre-saved MasterGenerators
    2. If not, processes raw dataframes:
        - Applies filters and recalculates labels
        - Splits data into train/val/test sets
        - Creates and saves new MasterGenerators

    Args:
        params: TrainingParameters object containing configuration settings
            including all parameters required for data processing and generator creation
        data: Dictionary containing either:
            - Pre-saved MasterGenerators with keys TRAIN_MASTER, VAL_MASTER, TEST_MASTER
            - Raw dataframes to be processed into generators

    Returns:
        TrainingMasterGenerators: A dataclass containing:
            - train: MasterGenerator for training data
            - val: MasterGenerator for validation data
            - test: MasterGenerator for test data
            - path: Path where generators are saved

    Raises:
        ValueError: If data dictionary format is invalid
        ValueError: If required parameters in TrainingParameters are missing
    """
    # Check if the loaded data contains ready MasterGenerators
    if isinstance(data, dict) and all(key in data for key in MASTER_GENERATOR_KEYS):
        logger.info("Found pre-saved MasterGenerators")
        return TrainingMasterGenerators(
            train=data[TRAIN_MASTER],
            val=data[VAL_MASTER],
            test=data[TEST_MASTER],
            params=data[PARAMS] if PARAMS in data.keys() else None,
            path=params.data_file
        )

    # Otherwise process as dataframes
    logger.info(f"{len(data)} dataframes loaded, preparing generators")

    if params.limit is not None and params.limit > 0:
        data = dict(list(data.items())[: params.limit])

    # Calculate filters and recalculate labels
    logger.info("Creating filter labels")
    calculate_filters(
        dataframes_dict=data,
        window_size=params.sequence_length,
        min_liquidity=params.min_liquidity,
        trading_volume_threshold=params.min_trading_volume,
        range_percent=params.min_price_range_percent,
        reverse=params.reverse_volume_filter,
        filter_column=params.filter_column,
    )

    logger.info("Recalculating labels")
    recalculate_labels(
        dataframes_dict=data,
        forward_length=params.forward_length,
        skip_length=params.skip_length,
        up_more_than=params.up_more_than,
        down_less_than=params.down_less_than,
        min_liquidity=params.min_liquidity,
        label_column=params.label,
    )

    # Split data and create generators
    train_dfs, val_dfs, test_dfs = prepare_train_val_test_split(
        dataframes_dict=data,
        test_hours=params.test_hours,
        val_hours=params.val_hours,
    )

    # Create generators
    train_master, val_master, test_master = create_master_generators(
        params=params,
        train_dfs=train_dfs,
        val_dfs=val_dfs,
        test_dfs=test_dfs
    )

    # Auto-save the created generators
    file_path = params.data_file
    base, ext = os.path.splitext(file_path)
    save_path = f"{base}{MASTER_GENERATOR_SUFFIX}{ext}" if not base.endswith(MASTER_GENERATOR_SUFFIX) else file_path

    gens = TrainingMasterGenerators(
        train=train_master,
        val=val_master,
        test=test_master,
        path=save_path,
        params=params,
    )

    logger.info("Auto-saving created MasterGenerators")
    save_master_generators(gens=gens)

    return gens

def save_master_generators(gens: TrainingMasterGenerators) -> None:
    """
    Save train, validation and test MasterGenerators to a pickle file.

    This function saves the generators in a format that can be loaded by
    load_master_generators() for future use, avoiding the need to recreate
    them from raw data.

    Args:
        gens: TrainingMasterGenerators dataclass containing:
            - train: MasterGenerator for training data
            - val: MasterGenerator for validation data
            - test: MasterGenerator for test data
            - path: Path where to save the pickle file

    Raises:
        PermissionError: If write permission is denied for the target path
        OSError: If the directory doesn't exist or other I/O errors occur
    """
    logger.info(f"Saving master generators to {gens.path}")
    generators = {
        TRAIN_MASTER: gens.train,
        VAL_MASTER: gens.val,
        TEST_MASTER: gens.test,
        PARAMS: gens.params,
    }
    with open(gens.path, "wb") as file:
        pickle.dump(generators, file)
    logger.info("Master generators saved successfully")


def prepare_generators(
    params: TrainingParameters,
    train_master: Optional[MasterGenerator] = None,
    val_master: Optional[MasterGenerator] = None,
    test_master: Optional[MasterGenerator] = None,
) -> Tuple[MultiPersonTimeSeriesGenerator, MultiPersonTimeSeriesGenerator, MultiPersonTimeSeriesGenerator, Dict[str, Any]]:
    """
    Create data generators for training, validation, and testing.

    This function either uses provided MasterGenerators or loads/creates them
    from file, then creates MultiPersonTimeSeriesGenerator instances for actual
    training use.

    Args:
        params: TrainingParameters object containing:
            - batch_size: Size of batches to generate
            - All parameters required by load_master_generators() if masters not provided
        train_master: Optional pre-created MasterGenerator for training data
        val_master: Optional pre-created MasterGenerator for validation data
        test_master: Optional pre-created MasterGenerator for test data

    Returns:
        Tuple[MultiPersonTimeSeriesGenerator, MultiPersonTimeSeriesGenerator, MultiPersonTimeSeriesGenerator, Dict[str, Any]]:
            - train_gen: Generator for training batches
            - val_gen: Generator for validation batches
            - test_gen: Generator for test batches
            - split_info: Dictionary containing information about the data split

    Raises:
        ValueError: If batch_size is invalid
        FileNotFoundError: If loading from file and data file doesn't exist
    """
    logger.info("Creating data generators")

    # If master generators are provided, use them directly
    if train_master is not None and val_master is not None and test_master is not None:
        logger.info("Using provided master generators")
        return create_data_generators_from_split(
            train_master=train_master,
            val_master=val_master,
            test_master=test_master,
            batch_size=params.batch_size,
        )

    # Otherwise, load or create master generators from file
    logger.info("Loading master generators from file")
    gens = load_master_generators(params=params)

    # Create generators using master generators
    return create_data_generators_from_split(
        train_master=gens.train,
        val_master=gens.val,
        test_master=gens.test,
        batch_size=params.batch_size,
    )
