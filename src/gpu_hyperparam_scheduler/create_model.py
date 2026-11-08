"""
Neural network model creation module for time series analysis.
This module provides classes and functions to create and compile a hybrid CNN-LSTM-Transformer
model optimized for time series data processing with imbalanced classification.
"""

# pylint: disable=no-member,import-error,no-name-in-module
import hashlib
import json
import numpy as np
from typing import List, Any
from dataclasses import dataclass, asdict

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
    LayerNormalization,
    BatchNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Activation,
    Multiply,
    Concatenate,
    Add,
    Normalization,
)
from tensorflow.keras import Model


@dataclass
class ModelParameters:
    """Configuration parameters for the neural network model.
    Attributes:
        num_heads: Number of attention heads in transformer blocks
        head_size: Size of each attention head
        ff_dim: Feed-forward network dimension in transformer blocks
        num_transformer_blocks: Number of transformer blocks in the model
        mlp_units: List of units in MLP layers
        dropout: Dropout rate for transformer blocks
        mlp_dropout: Dropout rate for MLP layers
        standard_normalizer: Whether to use standard normalization
        focal_gamma: Focusing parameter for focal loss
        focal_fp_weight: False positive weight for focal loss"""

    num_heads: int = 8
    head_size: int = 32
    ff_dim: int = 128
    num_transformer_blocks: int = 2
    mlp_units: List[int] = None
    dropout: float = 0.5
    mlp_dropout: float = 0.3
    standard_normalizer: bool = False
    focal_gamma: float = 2.5
    focal_fp_weight: float = 3.0

    def __post_init__(self):
        """Initialize default MLP units if none provided."""
        if self.mlp_units is None:
            self.mlp_units = [128, 64]


class ImbaFocalLoss(tf.keras.losses.Loss):
    """Custom focal loss implementation for imbalanced classification.
    This loss function combines focal loss with additional weighting for
    false positives to handle class imbalance."""

    def __init__(self, gamma=2.0, fp_weight=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.fp_weight = fp_weight

    def call(self, y_true, y_pred):
        """Calculate the imbalanced focal loss.
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
        Returns:
            Calculated loss value"""
        # Reshape y_true to match y_pred's shape
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        fp_mask = tf.cast((y_pred > 0.5) & (y_true == 0), tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        loss = focal_weight * bce * (1 + (self.fp_weight - 1) * fp_mask)
        return tf.reduce_mean(loss)

    def get_config(self):
        """Return the configuration for serialization."""
        return {**super().get_config(), "gamma": self.gamma, "fp_weight": self.fp_weight}


class MinMaxScalingLayer(tf.keras.layers.Layer):
    """Custom layer for min-max scaling of input features.
    This layer performs feature-wise min-max scaling to a specified range."""

    def __init__(self, feature_range=(0, 1), **kwargs):
        super().__init__(**kwargs)
        self.feature_range = feature_range

    def call(self, inputs):
        """Apply min-max scaling to input features.
        Args:
            inputs: Input tensor to be scaled
        Returns:
            Scaled tensor"""
        # Calculate min and max for each feature across the time dimension
        data_min = tf.reduce_min(inputs, axis=1, keepdims=True)
        data_max = tf.reduce_max(inputs, axis=1, keepdims=True)
        # Avoid division by zero
        data_range = tf.maximum(data_max - data_min, tf.keras.backend.epsilon())
        # Scale to [0, 1]
        scaled = (inputs - data_min) / data_range
        # Scale to feature range
        min_val, max_val = self.feature_range
        scaled = scaled * (max_val - min_val) + min_val
        return scaled

    def get_config(self):
        """Return the layer configuration for serialization.
        Returns:
            Dictionary containing the layer configuration"""
        config = super().get_config()
        config.update({"feature_range": self.feature_range})
        return config


def create_optimized_cnn_lstm_transformer(
    sequence_length: int = 200, n_features: int = 6, params: ModelParameters = None
) -> tf.keras.Model:
    """Create a hybrid CNN-LSTM-Transformer model for time series analysis.
    Args:
        sequence_length: Length of input time series sequences
        n_features: Number of features in input data
        params: Model configuration parameters
    Returns:
        Compiled Keras model"""
    if params is None:
        params = ModelParameters()

    inputs = Input(shape=(sequence_length, n_features))
    # Input normalization
    if params.standard_normalizer:
        # Standardization (mean=0, std=1)
        x = Normalization(axis=-1)(inputs)
    else:
        # MinMax scaling to [-1, 1] range
        x = MinMaxScalingLayer(feature_range=(-1, 1))(inputs)

    # First CNN block
    conv1 = Conv1D(filters=32, kernel_size=3, padding="same", activation=None)(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Dropout(0.2)(conv1)
    x = MaxPooling1D(pool_size=2)(conv1)

    # Second CNN block with residual
    conv2 = Conv1D(filters=64, kernel_size=3, padding="same", activation=None)(x)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Dropout(0.3)(conv2)
    res2 = Conv1D(filters=64, kernel_size=1)(x)
    x = Add()([conv2, res2])
    x = MaxPooling1D(pool_size=2)(x)

    # Transformer blocks
    for _ in range(params.num_transformer_blocks):
        attention_output = MultiHeadAttention(
            num_heads=params.num_heads,
            key_dim=params.head_size,
            dropout=params.dropout,
            attention_axes=(1,),
        )(x, x, x)
        # !!! epsilon=1e-5
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)
        ffn = Dense(params.ff_dim, activation="relu")(x)
        ffn = Dropout(params.dropout)(ffn)
        ffn = Dense(x.shape[-1])(ffn)
        x = LayerNormalization(epsilon=1e-6)(ffn + x)

    # Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(params.dropout)(x)

    # Enhanced attention mechanism
    # Self-attention layer
    self_attention = MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=params.dropout
    )(x, x, x)
    x = LayerNormalization(epsilon=1e-6)(self_attention + x)

    # Temporal attention
    attention_weights = Dense(x.shape[-1], activation="tanh")(x)
    attention_weights = Dense(1, activation="sigmoid")(attention_weights)
    weighted_output = Multiply()([x, attention_weights])

    # Multiple pooling strategies
    max_pooled = GlobalMaxPooling1D()(weighted_output)
    avg_pooled = GlobalAveragePooling1D()(weighted_output)

    # Combine pooled features
    x = Concatenate()([max_pooled, avg_pooled])

    # Streamlined MLP head
    for dim in params.mlp_units:
        x = Dense(dim)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(params.mlp_dropout)(x)

    x = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(
        x
    )
    probability = Dense(1, activation="sigmoid", name="probability")(x)

    return Model(inputs, probability)


def compile_model(
    model: tf.keras.Model, params: ModelParameters = None
) -> tf.keras.Model:
    """Compile the model with optimizer, loss function, and metrics.
    Args:
        model: Keras model to compile
        params: Model parameters including loss function settings
    Returns:
        Compiled Keras model"""
    if params is None:
        params = ModelParameters()

    # initial_learning_rate = 5e-4
    # decay_steps = 1000
    # decay_rate = 0.95

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=decay_steps,
    #     decay_rate=decay_rate,
    #     staircase=True
    # )

    optimizer = tf.keras.optimizers.Adam(
        # learning_rate=lr_schedule,
        clipnorm=0.5
    )

    loss = ImbaFocalLoss(gamma=params.focal_gamma, fp_weight=params.focal_fp_weight)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.Precision(name="precision", thresholds=0.75)],
    )

    return model


def get_callbacks(path: str) -> List[Any]:
    """Create a list of callbacks for model training.
    Args:
        path: Path to save model checkpoints
    Returns:
        List of Keras callbacks"""

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_precision", mode="max", patience=5, restore_best_weights=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_precision", mode="max", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path, monitor="val_precision", mode="max", save_best_only=False
        ),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=base, monitor="val_precision", mode="max", save_best_only=True
        # ),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=weights_path,
        #     monitor="val_precision", mode="max",
        #     save_best_only=True,
        #     save_weights_only=True,
        # ),
    ]
    return callbacks


def load_model(filepath: str, params: ModelParameters = None) -> tf.keras.Model:
    """Load a saved model with custom objects.
    Args:
        filepath: Path to saved model
    Returns:
        Loaded Keras model"""
    custom_objects = {
        "MinMaxScalingLayer": MinMaxScalingLayer,
        "ImbaFocalLoss": ImbaFocalLoss,
    }

    # Try loading as SavedModel first
    try:
        return tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    except:
        # If that fails, try creating a new model and loading weights
        print("Failed to load model directly. Attempting to load weights...")

        # Create a new model with default parameters
        model = create_optimized_cnn_lstm_transformer(params=params)

        # Try to load weights
        try:
            model.load_weights(filepath)
            return model
        except Exception as exc:
            raise Exception(
                f"Failed to load model from {filepath}. Please ensure the model was saved correctly."
            ) from exc

def convert_numpy(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

def get_config_id(params: ModelParameters) -> str:
    """Generate unique identifier for parameter configuration."""
    param_dict = asdict(params)
    # Convert numpy types to native Python types
    param_dict = convert_numpy(param_dict)
    param_str = json.dumps(param_dict, sort_keys=True)
    return hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()
