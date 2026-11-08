"""
Time series data generator functionality.
Implements sequence generation and batch processing for time series data.

Contains:
- MasterGenerator: Handles common data and indices shared across GPU instances
- MultiPersonTimeSeriesGenerator: Generator for creating batches using MasterGenerator
"""

import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Features

# Initialize module logger
logger = logging.getLogger(__name__)


def process_chunk(
    chunk_data: List[Tuple[int, np.ndarray, np.ndarray]],
    window_size: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a chunk of person data to create indices, excluding filtered out windows.

    Args:
        chunk_data: List of tuples containing (person_idx, (labels, filters))
        window_size: Size of the sliding window
        step: Step size between windows

    Returns:
        Tuple containing arrays of indices for class 0 and class 1
    """
    chunk_class_0 = []
    chunk_class_1 = []

    for person_idx, (labels, filters) in chunk_data:
        # Create all possible start indices
        start_indices = np.arange(0, len(labels) - window_size, step)

        # Get filter values for the end of each window
        window_filters = filters[start_indices + window_size]

        # Only keep windows that pass the filter
        valid_indices = start_indices[window_filters]

        if len(valid_indices) > 0:
            # Get labels for valid windows
            window_labels = labels[valid_indices + window_size]
            person_indices = np.full_like(valid_indices, person_idx)

            # Split by class
            mask_0 = window_labels == 0
            mask_1 = ~mask_0

            if np.any(mask_0):
                chunk_class_0.append(
                    np.column_stack((person_indices[mask_0], valid_indices[mask_0]))
                )
            if np.any(mask_1):
                chunk_class_1.append(
                    np.column_stack((person_indices[mask_1], valid_indices[mask_1]))
                )

    return (
        np.vstack(chunk_class_0) if chunk_class_0 else np.empty((0, 2), dtype=int),
        np.vstack(chunk_class_1) if chunk_class_1 else np.empty((0, 2), dtype=int),
    )


class MasterGenerator:
    """
    Master generator that handles common data and indices shared across GPU instances.

    This class:
    1. Stores common data (features, labels, filters)
    2. Creates and stores class indices that are shared across instances
    3. Processes data in parallel for efficiency

    Attributes:
        dataframes_list: List of DataFrames containing person data
        feature_columns: List of column names to use as features
        label_column: Name of the label column
        window_size: Size of the sliding window
        step: Step size between windows
    """

    def __init__(
        self,
        dataframes_list: List[pd.DataFrame],
        feature_columns: List[str],
        window_size: int,
        step: int = 1,
        label_column: str = Features.Y_CLASS.value,
        filter_column: str = Features.FILTER.value,
    ):
        self.dataframes_list = dataframes_list
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.window_size = window_size
        self.filter_column = filter_column
        self.step = step

        # Initialize storage for data
        self.features_list = []
        self.labels_list = []
        self.filters_list = []  # Store filter values

        self.class_0_indices = []
        self.class_1_indices = []

        # Process each dataframe separately
        for df in dataframes_list:
            self.features_list.append(df[feature_columns].values)
            self.labels_list.append(df[label_column].values)
            self.filters_list.append(df[filter_column].values)

        # Create indices for batch generation
        self._create_indices()

    def _create_indices(self):
        """Create list of valid indices for batch generation using parallel processing"""
        # Prepare data
        person_data = [
            (idx, (labels, filters))
            for idx, (labels, filters) in enumerate(
                zip(self.labels_list, self.filters_list)
            )
        ]

        # Split data into chunks for parallel processing
        num_cores = max(1, min(len(person_data), cpu_count() - 1))
        chunksize = max(1, len(person_data) // num_cores)
        chunks = [
            person_data[i : i + chunksize]
            for i in range(0, len(person_data), chunksize)
        ]

        # Process chunks in parallel
        process_func = partial(process_chunk, window_size=self.window_size, step=self.step)
        if num_cores == 1:
            results = [process_func(chunk) for chunk in chunks]
        else:
            with Pool(num_cores) as pool:
                results = pool.map(process_func, chunks)

        # Combine results
        class_0_indices = []
        class_1_indices = []
        for c0, c1 in results:
            if len(c0) > 0:
                class_0_indices.append(c0)
            if len(c1) > 0:
                class_1_indices.append(c1)

        # Store full indices for both classes
        self.class_0_indices = (
            np.vstack(class_0_indices)
            if class_0_indices
            else np.empty((0, 2), dtype=int)
        )
        self.class_1_indices = (
            np.vstack(class_1_indices)
            if class_1_indices
            else np.empty((0, 2), dtype=int)
        )

        logger.info(
            "\nTotal Windows Found:\n"
            "==================================================\n"
            "  Class 0:  %s\n"
            "  Class 1:  %s\n"
            "==================================================",
            f"{len(self.class_0_indices):,}",
            f"{len(self.class_1_indices):,}",
        )


class MultiPersonTimeSeriesGenerator(tf.keras.utils.Sequence):  # type: ignore
    """
    Generator for creating batches of time series data using MasterGenerator.
    Handles balanced sampling and batch creation.

    This generator:
    1. Uses shared data from MasterGenerator
    2. Supports balanced class sampling
    3. Creates unique batches for each GPU instance
    4. Manages epoch-level data shuffling

    Attributes:
        master: MasterGenerator instance containing shared data
        batch_size: Number of samples per batch
        balanced: Whether to balance classes in batches
    """

    def __init__(
        self,
        master: MasterGenerator,
        batch_size: int,
        balanced: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.master = master
        self.batch_size = batch_size
        self.balanced = balanced
        self.indices = None  # Initialize indices attribute

        # Initial fill of indices
        self._fill_indices(print_info=True)

        # Adjust batch size if needed
        if self.balanced and self.batch_size % 2 != 0:
            self.batch_size = self.batch_size + 1
            logger.info("Adjusted batch size to %d to ensure balance", self.batch_size)

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        """Get batch at position idx"""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_data = []
        batch_labels = []

        for person_idx, start_idx in batch_indices:
            # Get window of features
            window = self.master.features_list[person_idx][
                start_idx : start_idx + self.master.window_size
            ]

            if np.any(np.isnan(window)) or np.any(np.isinf(window)):
                raise ValueError(f"Invalid values detected in data: person_idx={person_idx}, start_idx={start_idx}")

            # Get label right after the window
            label = self.master.labels_list[person_idx][start_idx + self.master.window_size]

            batch_data.append(window)
            batch_labels.append(label)

        return np.array(batch_data), np.array(batch_labels)

    def get_labels(self, idx):
        """Get labels for a specific batch index"""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_labels = []

        for person_idx, start_idx in batch_indices:
            # Get label right after the window
            label = self.master.labels_list[person_idx][start_idx + self.master.window_size]
            batch_labels.append(label)

        return np.array(batch_labels)

    def _fill_indices(self, print_info: bool = False):
        """Fill indices with random samples for current epoch"""
        # Use all samples
        epoch_indices_0 = self.master.class_0_indices
        epoch_indices_1 = self.master.class_1_indices

        if self.balanced:
            # Get size of minority class
            min_class_size = min(len(self.master.class_0_indices), len(self.master.class_1_indices))

            # Randomly sample from both classes
            indices_0 = np.random.choice(
                len(self.master.class_0_indices), size=min_class_size, replace=False
            )
            indices_1 = np.random.choice(
                len(self.master.class_1_indices), size=min_class_size, replace=False
            )

            # Get selected samples
            epoch_indices_0 = self.master.class_0_indices[indices_0]
            epoch_indices_1 = self.master.class_1_indices[indices_1]

            if print_info:
                logger.info(
                    "Epoch Samples:\n"
                    "==================================================\n"
                    "  Class 0:  %s\n"
                    "  Class 1:  %s\n"
                    "==================================================",
                    f"{len(epoch_indices_0):,}",
                    f"{len(epoch_indices_1):,}",
                )

        # Combine and shuffle indices for this epoch
        self.indices = np.concatenate([epoch_indices_0, epoch_indices_1])
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        """Select new random samples at the end of every epoch"""
        self._fill_indices()
