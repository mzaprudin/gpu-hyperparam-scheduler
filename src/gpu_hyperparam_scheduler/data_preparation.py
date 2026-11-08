"""
Data preparation and processing functionality.
Implements data filtering, labeling, and splitting for time series data.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from enum import Enum, auto

from gpu_hyperparam_scheduler.data_generator import MultiPersonTimeSeriesGenerator, MasterGenerator
from gpu_hyperparam_scheduler.config import Features

# Initialize module logger
logger = logging.getLogger(__name__)

# Data column names and types
EPSILON = 1e-14

class LabelType(Enum):
    """Enum for different types of label calculations"""
    UP = auto()
    DOWN = auto()
    LIQ = auto()


def format_timestamp(unix_timestamp: int) -> str:
    """Convert Unix timestamp to readable format for display purposes only"""
    return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")


def prepare_train_val_test_split(
    dataframes_dict: Dict[str, pd.DataFrame],
    val_hours: int = 96,
    test_hours: int = 48
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split each person's data into train, validation and test based on Unix time,
    using the latest timestamp across all dataframes

    Args:
        dataframes_dict: dictionary where keys are person IDs and values are their dataframes
        val_hours: number of hours to use for validation before test data
        test_hours: number of hours to use for testing from the end of each dataset

    Returns:
        train_dfs: list of training dataframes
        val_dfs: list of validation dataframes
        test_dfs: list of testing dataframes
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Convert hours to seconds
    val_seconds = val_hours * 3600
    test_seconds = test_hours * 3600

    # Find the global latest timestamp
    latest_timestamp = max(df[Features.TIMESTAMP.value].max() for df in dataframes_dict.values())

    # Calculate split timestamps based on global latest timestamp
    test_split_timestamp = latest_timestamp - test_seconds
    val_split_timestamp = test_split_timestamp - val_seconds

    logger.info(
        f"\nTime Split Information:"
        f"\n{'='*50}"
        f"\nTimestamps:"
        f"\n  Latest:    {format_timestamp(latest_timestamp)}"
        f"\n  Test:      {format_timestamp(test_split_timestamp)}"
        f"\n  Val:       {format_timestamp(val_split_timestamp)}"
        f"\n  Duration:  Test: {test_hours} hours ({test_seconds:,} seconds)"
        f"\n            Val:  {val_hours} hours ({val_seconds:,} seconds)"
        f"\n{'='*50}"
    )

    # Split each person's data
    for person_id, df in dataframes_dict.items():
        # Sort by timestamp
        df = df.sort_values("block_number").reset_index(drop=True)

        # Split data into train, validation and test
        train_df = df[df[Features.TIMESTAMP.value] < val_split_timestamp].copy()
        val_df = df[(df[Features.TIMESTAMP.value] >= val_split_timestamp) &
                    (df[Features.TIMESTAMP.value] < test_split_timestamp)].copy()
        test_df = df[df[Features.TIMESTAMP.value] >= test_split_timestamp].copy()

        if not train_df.empty:
            train_dfs.append(train_df)
        if not val_df.empty:
            val_dfs.append(val_df)
        if not test_df.empty:
            test_dfs.append(test_df)

    # Overall statistics
    total_train_samples = sum(len(df) for df in train_dfs)
    total_val_samples = sum(len(df) for df in val_dfs)
    total_test_samples = sum(len(df) for df in test_dfs)

    logger.info(
        f"\nSplit Statistics:"
        f"\n{'='*50}"
        f"\nSample Counts:"
        f"\n  Training:    {total_train_samples:,}"
        f"\n  Validation:  {total_val_samples:,}"
        f"\n  Test:        {total_test_samples:,}"
        f"\n{'='*50}"
    )

    if train_dfs:
        train_start = min(df[Features.TIMESTAMP.value].min() for df in train_dfs)
        train_end = max(df[Features.TIMESTAMP.value].max() for df in train_dfs)
        logger.info(
            f"\nTraining Period:"
            f"\n{'='*50}"
            f"\n  Start:     {format_timestamp(train_start)}"
            f"\n  End:       {format_timestamp(train_end)}"
            f"\n  Duration:  {(train_end - train_start) / 3600:.2f} hours"
            f"\n{'='*50}"
        )

    if val_dfs:
        val_start = min(df[Features.TIMESTAMP.value].min() for df in val_dfs)
        val_end = max(df[Features.TIMESTAMP.value].max() for df in val_dfs)
        logger.info(
            f"\nValidation Period:"
            f"\n{'='*50}"
            f"\n  Start:     {format_timestamp(val_start)}"
            f"\n  End:       {format_timestamp(val_end)}"
            f"\n  Duration:  {(val_end - val_start) / 3600:.2f} hours"
            f"\n{'='*50}"
        )

    if test_dfs:
        test_start = min(df[Features.TIMESTAMP.value].min() for df in test_dfs)
        test_end = max(df[Features.TIMESTAMP.value].max() for df in test_dfs)
        logger.info(
            f"\nTest Period:"
            f"\n{'='*50}"
            f"\n  Start:     {format_timestamp(test_start)}"
            f"\n  End:       {format_timestamp(test_end)}"
            f"\n  Duration:  {(test_end - test_start) / 3600:.2f} hours"
            f"\n{'='*50}"
        )

    if not train_dfs:
        raise ValueError("No training data available after splitting")
    if not val_dfs:
        raise ValueError("No validation data available after splitting")
    if not test_dfs:
        raise ValueError("No test data available after splitting")

    return train_dfs, val_dfs, test_dfs


def create_data_generators_from_split(
    train_master: MasterGenerator,
    val_master: MasterGenerator,
    test_master: MasterGenerator,
    batch_size: int = 1024,
) -> Tuple[
    MultiPersonTimeSeriesGenerator,
    MultiPersonTimeSeriesGenerator,
    MultiPersonTimeSeriesGenerator,
    Dict[str, Any],
]:
    """
    Create train, validation and test generators from pre-created MasterGenerator instances.
    Note: Filters and labels should be pre-calculated before calling this function.
    """
    logger.info("Create generators")
    # Create generators
    train_gen = MultiPersonTimeSeriesGenerator(
        master=train_master,
        batch_size=batch_size,
        balanced=True,
    )

    val_gen = MultiPersonTimeSeriesGenerator(
        master=val_master,
        batch_size=batch_size,
        balanced=False,
    )

    test_gen = MultiPersonTimeSeriesGenerator(
        master=test_master,
        batch_size=batch_size,
        balanced=False,
    )

    def calculate_class_dist(master: MasterGenerator) -> Tuple[int, float]:
        """Calculate total samples and class distribution from MasterGenerator indices"""
        total = len(master.class_0_indices) + len(master.class_1_indices)
        if total == 0:
            return 0, 0.0

        class_dist = len(master.class_1_indices) / total
        return total, class_dist

    # Collect split info
    train_samples, train_dist = calculate_class_dist(train_master)
    val_samples, val_dist = calculate_class_dist(val_master)
    test_samples, test_dist = calculate_class_dist(test_master)

    split_info = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_class_dist": train_dist,
        "val_class_dist": val_dist,
        "test_class_dist": test_dist,
    }

    logger.info(
        f"\nData Split Information:"
        f"\n{'='*50}"
        f"\nSamples:"
        f"\n  Training:     {split_info['train_samples']:,}"
        f"\n  Validation:   {split_info['val_samples']:,}"
        f"\n  Test:         {split_info['test_samples']:,}"
        f"\n{'-'*50}"
        f"\nClass 1 distribution:"
        f"\n  Training:     {split_info['train_class_dist']:.2%}"
        f"\n  Validation:   {split_info['val_class_dist']:.2%}"
        f"\n  Test:         {split_info['test_class_dist']:.2%}"
        f"\n{'='*50}"
    )

    return train_gen, val_gen, test_gen, split_info


def create_activity_filter(
    df: pd.DataFrame,
    window_size: int,
    min_liquidity: float,
    trading_volume_threshold: float,
    range_percent: float = 0.1,
    reverse: bool = False,
    filter_column: str = Features.FILTER.value,
) -> pd.DataFrame:
    """
    Create filter column using vectorized operations.
    The filter checks three conditions:
    1. Rolling sum of absolute values of Features.BUY_VOLUME and Features.SELL_VOLUME compared to `threshold`.
    2. Rolling window max / min > `range_threshold`.
    3. Last value of Features.LIQUIDITY_SOL in the rolling window >= `min_liquidity`.

    Args:
        df: DataFrame containing required columns (BUY_VOLUME, SELL_VOLUME, PRICE, LIQUIDITY_SOL)
        window_size: Size of the rolling window for calculations
        min_liquidity: Minimum required liquidity value at the end of each window
        trading_volume_threshold: Threshold for comparing rolling sum of trading volumes
        range_percent: Percentage for calculating price range threshold (default: 0.1)
        reverse: If True, inverts the volume threshold comparison (default: False)
        filter_column: Name of the output filter column (default: Features.FILTER)

    Returns:
        DataFrame with added filter column containing boolean values
    """
    # Handle NaN values
    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        df.dropna(inplace=True)
        print("NaN detected")

    # Extract values
    b_values = df[Features.BUY_VOLUME.value].abs().values
    s_values = df[Features.SELL_VOLUME.value].abs().values
    n_rows = len(b_values)

    if n_rows < window_size:
        df[filter_column] = False
        return df

    # Create windows view for both arrays
    b_windows = np.lib.stride_tricks.sliding_window_view(b_values, window_size)
    s_windows = np.lib.stride_tricks.sliding_window_view(s_values, window_size)

    # Calculate rolling sums using numpy operations
    rolling_sum = np.sum(b_windows, axis=1) + np.sum(s_windows, axis=1)

    # Calculate rolling max/min using P_i values
    p_values = df[Features.PRICE.value].values
    p_windows = np.lib.stride_tricks.sliding_window_view(p_values, window_size)
    rolling_max = np.maximum.reduce(p_windows, axis=1)
    rolling_min = np.minimum.reduce(p_windows, axis=1)

    # Calculate range and threshold
    rolling_range = rolling_max - rolling_min
    range_threshold = range_percent * rolling_max

    # Get liquidity values and create window view
    l_values = df[Features.LIQUIDITY_SOL.value].values
    l_windows = np.lib.stride_tricks.sliding_window_view(l_values, window_size)
    last_liquidity = l_windows[:, -1]  # Get last value of each window

    # Create filter array
    filter_array = np.zeros(n_rows, dtype=bool)
    valid_indices = slice(0, len(rolling_sum))

    if reverse:
        filter_array[valid_indices] = (
            (rolling_sum <= trading_volume_threshold) &
            (rolling_range > range_threshold) &
            (last_liquidity >= min_liquidity)
        )
    else:
        filter_array[valid_indices] = (
            (rolling_sum >= trading_volume_threshold) &
            (rolling_range > range_threshold) &
            (last_liquidity >= min_liquidity)
        )

    df[filter_column] = filter_array
    return df


def calculate_filters(
    dataframes_dict: Dict[str, pd.DataFrame],
    window_size: int,
    min_liquidity: float,
    trading_volume_threshold: float,
    range_percent: float = 0.1,
    reverse: bool = False,
    filter_column: str = Features.FILTER.value,
) -> None:
    """
    Calculate filters for all dataframes in parallel using pool.map with chunksize
    """
    # Get dataframes and count
    dfs = dataframes_dict.values()
    n_dfs = len(dataframes_dict)

    # Calculate optimal number of processes and chunksize
    n_jobs = max(1, min(n_dfs, cpu_count() - 1))
    chunksize = max(1, n_dfs // (n_jobs * 4))  # Rule of thumb: chunks = 4 * n_jobs

    # Create partial function with fixed parameters
    proc_df = partial(
        create_activity_filter,
        window_size=window_size,
        min_liquidity=min_liquidity,
        trading_volume_threshold=trading_volume_threshold,
        range_percent=range_percent,
        reverse=reverse,
        filter_column=filter_column,
    )

    logger.info(f"Processing {n_dfs} dataframes using {n_jobs} processes")
    logger.info(f"Chunksize: {chunksize}")

    with Pool(n_jobs) as pool:
        processed_dfs = pool.map(proc_df, dfs, chunksize=chunksize)
    for person_id, df in zip(dataframes_dict.keys(), processed_dfs):
        dataframes_dict[person_id] = df

    # Print statistics
    total_samples = sum(len(df) for df in dataframes_dict.values())
    total_filtered = sum(df[filter_column].sum() for df in dataframes_dict.values())

    logger.info(
        f"\nFilter Statistics:"
        f"\n{'='*50}"
        f"\n  Total samples:        {total_samples:,}"
        f"\n  Passing filter:       {total_filtered:,} ({total_filtered/total_samples*100:.2f}%)"
        f"\n  Filtered out:         {total_samples-total_filtered:,} ({(total_samples-total_filtered)/total_samples*100:.2f}%)"
        f"\n{'='*50}"
    )


def calculate_rolling_labels(
    df: pd.DataFrame,
    window_length: int,
    skip_length: int,
    label_type: LabelType
) -> pd.Series:
    """
    Calculate rolling labels for time series data using vectorized operations.

    Args:
        df: DataFrame containing Features.PRICE column with price data
        window_length: Length of rolling window for calculations
        label_type: Type of label to calculate (UP or DOWN)

    Returns:
        pd.Series with calculated labels

    Raises:
        ValueError: If window_length is not positive
        KeyError: If Features.PRICE column is missing from DataFrame
    """
    # Validate inputs
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    required_column = Features.LIQUIDITY_SOL.value if label_type == LabelType.LIQ else Features.PRICE.value

    if required_column not in df.columns:
        raise KeyError(f"DataFrame must contain {required_column} column")

    # Handle empty DataFrame
    if len(df) == 0:
        return pd.Series([], index=df.index)

    # Extract P_i column values
    values = df[required_column].values
    n_rows = len(values)
    result = np.zeros(n_rows)

    if n_rows < window_length:
        return pd.Series(result, index=df.index)

    # Create windows view with adjusted length
    effective_window = window_length - skip_length
    windows = np.lib.stride_tricks.sliding_window_view(values, effective_window)

    # Calculate first values of each window
    first_values = windows[:, 0]

    # Handle zero/small first values
    valid_mask = first_values > EPSILON

    if label_type == LabelType.UP:
        # Vectorized up calculation
        max_values = np.maximum.reduce(windows, axis=1)
        high = np.zeros(len(windows))
        high[valid_mask] = np.maximum(EPSILON, max_values[valid_mask] / first_values[valid_mask] - 1)
        # Shift results backward by skip_length (e.g. value at index 15 goes to index 0)
        valid_length = len(high) - skip_length
        result[:valid_length] = high[skip_length:]
    elif label_type == LabelType.DOWN:
        # Vectorized down calculation
        min_values = np.minimum.reduce(windows, axis=1)
        low = np.zeros(len(windows))
        low[valid_mask] = np.maximum(EPSILON, 1 - min_values[valid_mask] / first_values[valid_mask])
        # Shift results backward by skip_length (e.g. value at index 15 goes to index 0)
        valid_length = len(low) - skip_length
        result[:valid_length] = low[skip_length:]
    elif label_type == LabelType.LIQ:
        # Vectorized liq calculation - get minimum value for each window
        min_values = np.minimum.reduce(windows, axis=1)
        # Shift results backward by skip_length (e.g. value at index 15 goes to index 0)
        valid_length = len(min_values) - skip_length
        result[:valid_length] = min_values[skip_length:]
    else:
        raise KeyError(f"Unsupported label type {label_type}")

    return pd.Series(result, index=df.index)


def process_single_df(
    df: pd.DataFrame,
    forward_length: int,
    skip_length: int,
    up_more_than: float,
    down_less_than: float,
    min_liquidity: float,
    label_column: str = Features.Y_CLASS.value,
    ) -> pd.DataFrame:
    """Process a single DataFrame to calculate labels using vectorized operations.

    Args:
        df: DataFrame to process
        forward_length: Number of periods to look forward
        up_more_than: Threshold for upward price movement
        down_less_than: Threshold for downward price movement

    Returns:
        Processed DataFrame with calculated labels
    """
    # Calculate y_up and y_down
    df[Features.Y_UP.value] = calculate_rolling_labels(
        df=df[[Features.PRICE.value]],
        window_length=forward_length,
        skip_length=skip_length,
        label_type=LabelType.UP
    )
    df[Features.Y_DN.value] = calculate_rolling_labels(
        df=df[[Features.PRICE.value]],
        window_length=forward_length,
        skip_length=skip_length,
        label_type=LabelType.DOWN
    )
    df[Features.Y_LIQ.value] = calculate_rolling_labels(
        df=df[[Features.LIQUIDITY_SOL.value]],
        window_length=forward_length,
        skip_length=skip_length,
        label_type=LabelType.LIQ
    )

    df.dropna(inplace=True)

    # Vectorized label calculation
    y_up_abs = np.abs(df[Features.Y_UP.value].values)
    y_down_abs = np.abs(df[Features.Y_DN.value].values)
    y_liq = np.abs(df[Features.Y_LIQ.value].values)

    if down_less_than > up_more_than:  # reverse case
        df[label_column] = ((y_up_abs < up_more_than) &
                        (y_down_abs > down_less_than) # & (y_liq > min_liquidity)
                        ).astype(np.int32)
    else:
        df[label_column] = ((y_up_abs > up_more_than) &
                        (y_down_abs < down_less_than) &
                        (y_liq > min_liquidity)).astype(np.int32)

    return df


def recalculate_labels(
    dataframes_dict: Dict[str, pd.DataFrame],
    up_more_than: float,
    down_less_than: float,
    forward_length: int,
    skip_length: int,
    min_liquidity: float,
    label_column: str,
) -> None:
    """
    Recalculate labels for each dataframe in the dictionary using parallel processing with pool.map chunksize.

    Args:
        dataframes_dict: Dictionary of dataframes
        up_more_than: Threshold for upward price movement
        down_less_than: Threshold for downward price movement
        forward_length: Number of periods to look forward for price movements
    """
    # Get dataframes and count
    dfs = dataframes_dict.values()
    n_dfs = len(dataframes_dict)

    # Calculate optimal number of processes and chunksize
    n_jobs = max(1, min(n_dfs, cpu_count() - 1))
    chunksize = max(1, n_dfs // (n_jobs * 4))  # Rule of thumb: chunks = 4 * n_jobs

    # Create partial function with fixed parameters
    proc_df = partial(
        process_single_df,
        forward_length=forward_length,
        skip_length=skip_length,
        up_more_than=up_more_than,
        down_less_than=down_less_than,
        min_liquidity=min_liquidity,
        label_column=label_column,
    )

    logger.info(f"Processing {n_dfs} dataframes using {n_jobs} processes")
    logger.info(f"Chunksize: {chunksize}")

    with Pool(n_jobs) as pool:
        processed_dfs = pool.map(proc_df, dfs, chunksize=chunksize)
    for person_id, df in zip(dataframes_dict.keys(), processed_dfs):
        dataframes_dict[person_id] = df

    # Calculate statistics
    total_samples = sum(len(df) for df in dataframes_dict.values())
    total_positive = sum(df[label_column].sum() for df in dataframes_dict.values())

    # Print overall statistics
    logger.info(
        f"\nLabel Statistics:"
        f"\n{'='*50}"
        f"\n  Total samples:     {total_samples:,}"
        f"\n  Positive samples:  {total_positive:,} ({total_positive/total_samples*100:.2f}%)"
        f"\n{'='*50}"
    )
