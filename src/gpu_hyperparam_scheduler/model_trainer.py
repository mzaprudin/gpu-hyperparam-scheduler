"""
Core model training and evaluation functionality.
"""

import json
import logging
import os
from typing import Optional, Tuple
from dataclasses import asdict
import tensorflow as tf

from gpu_hyperparam_scheduler.create_model import (
    create_optimized_cnn_lstm_transformer,
    compile_model,
    get_callbacks,
    load_model,
    ModelParameters,
)
from gpu_hyperparam_scheduler.config import TrainingParameters, ResultValues
from gpu_hyperparam_scheduler.data_generator import MasterGenerator
from gpu_hyperparam_scheduler.data_loader import prepare_generators
from gpu_hyperparam_scheduler.types import EvaluationMode
from gpu_hyperparam_scheduler.metrics import evaluate_model_performance, ModelEvalParams, ModelEvaluationResult
from gpu_hyperparam_scheduler.utils import TrainerError, set_seeds
from gpu_hyperparam_scheduler.visualization import analyze_probability_distribution, print_comparison_report, print_evaluation_report, PlotParameters

# Configure logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training, evaluation, and analysis of neural network models.

    This class manages the complete lifecycle of model training and evaluation:
    - GPU setup and memory management
    - Data loading and generator creation
    - Model training and evaluation
    - Performance analysis and visualization
    - Result saving and reporting

    Attributes:
        params: Training configuration parameters
        model_params: Model architecture parameters (set during optimization)
        train_gen: Training data generator
        val_gen: Validation data generator
        test_gen: Test data generator
        model: Current TensorFlow model
        history: Training history
        results: Evaluation results
    """

    def __init__(self, params: TrainingParameters, verbose: int = 1):
        """Initialize trainer with configuration parameters."""
        self.params = params
        self.verbose = verbose
        self.model_params = None  # Will be set during optimization
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.model = None
        self.history = None
        self.results = None
        self.optimal_threshold = None

        self.base_name = None
        self.file_name_model = None
        self.file_name_pic = None
        self.file_name_json = None
        self.file_name_weights = None

        self._setup_gpu()
        set_seeds(self.params.seed)

    def prepare(
        self,
        train_master: Optional[MasterGenerator] = None,
        val_master: Optional[MasterGenerator] = None,
        test_master: Optional[MasterGenerator] = None,
    ) -> None:
        """
        Load data and prepare data generators.

        Args:
            train_master: Optional pre-created MasterGenerator for training data
            val_master: Optional pre-created MasterGenerator for validation data
            test_master: Optional pre-created MasterGenerator for test data

        Loads training data and creates data generators for training,
        validation, and testing. If no generators are provided, loads them
        from the file specified in params.data_file (supports both pre-saved
        generators and raw dataframes). This must be called before training
        or evaluation.
        """
        logger.info("Creating data generators")
        generators = prepare_generators(
            params=self.params,
            train_master=train_master,
            val_master=val_master,
            test_master=test_master,
        )
        self.train_gen, self.val_gen, self.test_gen, _ = generators

    def evaluate(self, model_params: ModelParameters = None) -> ResultValues:
        """
        Train and evaluate model with given parameters.

        Args:
            model_params: Parameters for model architecture and training.
                      If None, uses default parameters.

        Returns:
            ResultValues containing evaluation metrics
        """
        # Clear resources first
        logger.debug("clear model")
        if self.model:
            del self.model
        logger.debug("clear history")
        if self.history:
            del self.history
        logger.debug("clear results")
        if self.results:
            del self.results

        # Clear session (but don't reset graph to maintain GPU visibility)
        logger.debug("clear session")
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        visible_devices = tf.config.get_visible_devices("GPU")
        if visible_devices:
            # device_name = visible_devices[0].name.replace("/physical_device:", "")
            tf.config.experimental.reset_memory_stats("GPU:0")

        self.model_params = model_params or ModelParameters(
            standard_normalizer=self.params.standard_normalizer
        )

        self.results = None

        self._generate_filenames()  # Generate filenames after model_params is set
        self.model, self.history = self._train_model()
        # Determine optimal threshold on validation data

        logger.info("Finding optimal threshold on validation data")
        val_eval_params = ModelEvalParams(
            test_gen=self.val_gen,
            model=self.model,
            min_threshold=self.params.min_threshold,
            threshold_step=self.params.threshold_step,
            min_tp_ratio=self.params.min_tp_ratio,
            verbose=self.verbose,
            evaluation_mode=EvaluationMode.EVALUATE
        )
        val_result = evaluate_model_performance(val_eval_params)
        self.optimal_threshold = val_result.threshold_results.optimal_threshold
        logger.info("Optimal threshold from validation: %f", self.optimal_threshold)

        self.results = self._analyze_test_performance(forced_threshold=self.optimal_threshold)

        print_comparison_report(evaluation=val_result.metrics, test=self.results)

        return self.results

    def test(self, model_path: str = None, name: str = None) -> ResultValues:
        """
        Test an existing model without training.

        Args:
            model_path: Path to the model file to load
            name: Optional name for results files. If None, uses model filename.

        Returns:
            ResultValues containing evaluation metrics
        """
        if model_path is None:
            model_path = self.params.model_to_load

        logger.info(f"Loading model from: {model_path}")
        self.model = load_model(filepath=model_path)

        # Set name for result files
        if name:
            self.params.name = name
        else:
            # Use model filename without extension
            self.params.name = os.path.basename(model_path).rsplit(".", 1)[0]

        self._generate_filenames()
        self.results = self._analyze_test_performance()

        print_evaluation_report(metrics=self.results)

        return self.results

    def _setup_gpu(self) -> None:
        """
        Configure GPU settings for TensorFlow.

        Sets up GPU visibility and memory growth settings based on the
        configured GPU index. Enables dynamic memory growth to prevent
        TensorFlow from allocating all GPU memory at startup.
        """
        logger.info(f"TensorFlow version: {tf.__version__}")

        if self.params.gpu_idx == -1:
            tf.config.set_visible_devices([], "GPU")
            logger.info("Using CPU")
            return
        gpus = tf.config.list_physical_devices("GPU")

        try:
            if not gpus:
                raise TrainerError("No GPU devices found. Use --cpu for training on this machine.")

            if self.params.gpu_idx >= len(gpus):
                raise TrainerError(
                    f"GPU index {self.params.gpu_idx} is out of range. Available GPUs: {len(gpus)}"
                )

            # Use specific GPU
            gpu = gpus[self.params.gpu_idx]
            logger.info(f"Using GPU: {gpu}")

            # Configure visible devices before any other GPU operations
            tf.config.set_visible_devices(gpu, "GPU")
            visible_devices = tf.config.get_visible_devices("GPU")
            if not visible_devices:
                raise TrainerError(
                    f"Failed to set GPU {self.params.gpu_idx} as visible device"
                )

            # Configure memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU dynamic memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"Error setting memory growth: {e}")
                # Continue even if memory growth setup fails
                pass

        except Exception as e:
            raise TrainerError(f"GPU setup failed: {str(e)}")

    def _generate_filenames(self) -> None:
        """Generate unique filenames based on parameters"""
        # Use name if provided, otherwise generate from parameters
        if self.params.name:
            base_name = self.params.name
        else:
            # Base name components
            name_parts = [
                f"model_{self.params.gpu_idx}",
                f"{self.params.min_tp_ratio}",
                f"{self.params.min_trading_volume}",
                f"{self.params.up_more_than}",
                f"{self.params.down_less_than}",
                f"{self.params.reverse_volume_filter}",
            ]

            # Add model architecture components if available
            if self.model_params:
                model_parts = [
                    f"h{self.model_params.num_heads}",
                    f"hs{self.model_params.head_size}",
                    f"tb{self.model_params.num_transformer_blocks}",
                    f"g{self.model_params.focal_gamma:.1f}",
                    f"fp{self.model_params.focal_fp_weight:.1f}",
                ]
                name_parts.extend(model_parts)

            base_name = "_".join(str(part) for part in name_parts)

        os.makedirs(self.params.save_dir, exist_ok=True)

        # Generate full paths
        self.base_path = os.path.join(self.params.save_dir, base_name)
        self.file_name_model = os.path.join(self.params.save_dir, f"{base_name}.keras")
        self.file_name_pic = os.path.join(self.params.save_dir, f"{base_name}.png")
        self.file_name_json = os.path.join(self.params.save_dir, f"{base_name}.json")
        self.file_name_weights = os.path.join(self.params.save_dir, f"{base_name}.w")

    def _train_model(self) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the model using provided data generators.

        Creates or loads a model based on configuration, then trains it using
        the training and validation data generators. Uses early stopping and
        model checkpointing callbacks to save the best model during training.

        Returns:
            Tuple containing:
            - Trained TensorFlow model
            - Training history with metrics for each epoch
        """
        # Compile model
        if self.params.model_to_load is None:
            # Create model
            logger.info("Creating optimized CNN-LSTM-Transformer model")
            model = create_optimized_cnn_lstm_transformer(
                sequence_length=self.train_gen.master.window_size,
                n_features=len(self.train_gen.master.feature_columns),
                params=self.model_params,
            )
            logger.info("Compiling model")
            model = compile_model(model=model, params=self.model_params)
        else:
            logger.info("Loading pre-trained model")
            model = load_model(filepath=self.params.model_to_load)

        callbacks = get_callbacks(path=self.file_name_model)

        # Train model
        logger.info("Starting model training")
        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.params.num_epochs,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        # base = os.path.splitext(self.file_name_model)[0]
        # model_path = base + ".1.keras"
        # logger.info("Save model to %s", model_path)
        # model.save(filepath=model_path)
        # logger.info("Export model to %s", base)
        # model.export(base)

        logger.info("Training completed")

        logger.info("Try to reload model from %s checkpoint", self.file_name_model)
        model = load_model(self.file_name_model)

        return model, history

    def _analyze_test_performance(self, forced_threshold: float = None) -> ResultValues:
        """
        Perform complete model evaluation and analysis.

        Evaluates model performance on test data, generates visualizations,
        and saves results to files. This includes:
        - Computing metrics using optimal threshold
        - Generating probability distribution plots
        - Saving results to JSON file

        Returns:
            ResultValues containing all computed metrics
        """

        threshold = forced_threshold if forced_threshold is not None else self.params.min_threshold
        mode = EvaluationMode.TEST if forced_threshold is not None or self.params.test_only else EvaluationMode.EVALUATE

        # Get evaluation results and predictions
        eval_params = ModelEvalParams(
            test_gen=self.test_gen,
            model=self.model,
            min_threshold=threshold,
            threshold_step=self.params.threshold_step,
            min_tp_ratio=self.params.min_tp_ratio,
            verbose=self.verbose,
            evaluation_mode=mode
        )
        eval_result = evaluate_model_performance(eval_params)
        self.results = eval_result.metrics

        # Analyze probability distribution using saved predictions
        if self.file_name_pic:
            plot_params = PlotParameters(
                probabilities=eval_result.probabilities,
                true_labels=eval_result.true_labels,
                optimals=eval_result.threshold_results,
                params=self.params,
                model_params=self.model_params,
                results=self.results,
                file_name_pic=self.file_name_pic,
            )
            analyze_probability_distribution(plot_params)

        if self.file_name_json:
            # Combine results with model and training parameters using dataclass serialization
            full_results = {
                "results": asdict(self.results),
                "model_parameters": asdict(self.model_params) if self.model_params else {},
                "training_parameters": asdict(self.params)
            }
            with open(self.file_name_json, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=4)
            logger.info(f"Results saved to: {self.file_name_json}")

        # if self.params.save_as_folder and self.base_path is not None:
        #     self.model.save_model(self.base_path)

        return self.results
