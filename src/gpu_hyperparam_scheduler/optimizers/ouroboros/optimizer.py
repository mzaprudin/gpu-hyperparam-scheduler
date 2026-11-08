"""
Ouroboros Optimizer: A hybrid optimization approach combining gradient descent,
binary search, and historical knowledge integration.

The Ouroboros, an ancient symbol of a serpent eating its own tail, represents the
cyclical nature of this optimizer where results feed back into the optimization process.
This creates a self-improving system that becomes more efficient over time.
"""

import json
import logging
import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
import time
from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from gpu_hyperparam_scheduler.param_range import ParamType
from gpu_hyperparam_scheduler.types import OptimizationTask, OptimizationResult, TypedQueue, WorkerConfig
from gpu_hyperparam_scheduler.parameter_sampler import ParameterSampler
from gpu_hyperparam_scheduler.hyperparameters import ModelHyperparameters
from gpu_hyperparam_scheduler.create_model import ModelParameters, get_config_id
from gpu_hyperparam_scheduler.config import TrainingParameters

from .config import (
    OuroborosConfig,
    OuroborosInitConfig,
    RESULT_TIMEOUT,
    MAX_SENSITIVE_PARAMS,
)
from .knowledge_nexus import KnowledgeNexus, SortedParameter
from .worker_manager import WorkerManager
from .parameter_optimizer import ParameterOptimizer
from .parameter_validator import ParameterValidator
from .task_manager import TaskManager


class TaskType(Enum):
    """Types of optimization tasks."""

    INITIAL = "initial"
    GRADIENT = "gradient"
    BINARY = "binary"
    MUTATION = "mutation"  # Genetic algorithm mutation task


class OuroborosOptimizer:
    """
    Ouroboros Optimizer: Combines gradient-based exploration, binary search, and
    historical knowledge integration for efficient hyperparameter optimization.
    """

    @property
    def best_precision(self) -> float:
        return self.task_manager.best_precision

    def __init__(self, init_config: OuroborosInitConfig):
        """Initialize the Ouroboros Optimizer."""
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Store configuration
        self._validate_init_config(init_config)
        self.training_params: TrainingParameters = init_config.training_params
        self.config: OuroborosConfig = init_config.config or OuroborosConfig()
        self.genetic_only = init_config.genetic_only
        self.results_path = Path(self.training_params.save_dir) / "optimization_results.json"
        self.gpu_indices: List[int] = init_config.gpu_indices or [0]
        self.num_gpus: int = len(self.gpu_indices)

        # Initialize multiprocessing components
        self.mp_ctx: mp.context.BaseContext = mp.get_context(init_config.mp_context)
        self.task_queue: TypedQueue[Optional[OptimizationTask]] = TypedQueue(
            self.mp_ctx
        )
        self.result_queue: TypedQueue[Optional[OptimizationResult]] = TypedQueue(
            self.mp_ctx
        )

        # Initialize optimization components
        self.hp: ModelHyperparameters = ModelHyperparameters()
        self.hp.set_standard_normalizer(self.training_params.standard_normalizer)
        self.sampler: ParameterSampler = ParameterSampler(self.hp)
        self.start_time: float = time.time()
        self.knowledge: KnowledgeNexus = KnowledgeNexus()

        # Add tracking for parameter exploration
        self.last_explored_params: Dict[str, float] = {}  # param_name -> last exploration time
        self.param_exploration_order: List[str] = list(self.hp.param_ranges.keys())

        try:
            # Initialize worker manager

            worker_base_config: WorkerConfig = WorkerConfig(
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                base_params=self.training_params,
                train_master=init_config.train_master,
                val_master=init_config.val_master,
                test_master=init_config.test_master,
            )

            self.worker_manager: WorkerManager = WorkerManager(
                gpu_indices=self.gpu_indices,
                mp_ctx=self.mp_ctx,
                worker_base_config=worker_base_config
            )

            # Initialize task manager
            self.task_manager: TaskManager = TaskManager(
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                knowledge=self.knowledge,
                gpu_indices=self.gpu_indices,
            )

            # Initialize parameter optimizer
            self.param_optimizer: ParameterOptimizer = ParameterOptimizer(
                hyperparameters=self.hp,
                task_manager=self.task_manager,
                min_epochs=self.config.min_epochs,
            )

            # Initialize parameter validator
            self.validator: ParameterValidator = ParameterValidator(self.logger)

            # Load previous results
            self._load_results()

        except Exception as e:
            self.logger.error("Failed to initialize optimizer: %s", str(e))
            self.cleanup()
            raise

    def _validate_init_config(self, init_config: OuroborosInitConfig) -> None:
        """Validate initialization configuration."""
        if not isinstance(init_config.gpu_indices, list) or not all(
            isinstance(idx, int) for idx in init_config.gpu_indices
        ):
            raise ValueError("gpu_indices must be a list of integers")
        if len(init_config.gpu_indices) != len(set(init_config.gpu_indices)):
            raise ValueError("gpu_indices contains duplicate values")

    def _load_results(self) -> None:
        """Load and restore previous optimization results."""
        if not self.results_path.exists():
            return

        try:
            with self.results_path.open("r", encoding="utf-8") as f:
                results: Dict[str, Any] = json.load(f)

            # Restore best parameters and precision
            if "best_params" in results:
                params: Dict[str, Any] = results["best_params"]
                if self.validator.validate_parameter_values(
                    params=params,
                    param_ranges=self.hp.param_ranges
                ):
                    self.task_manager.best_params = ModelParameters(**params)
                    self.task_manager.best_precision = float(results["best_precision"])
                else:
                    self.logger.warning("Skipping invalid best parameters")

            # Restore results to knowledge base
            valid_results: int = 0
            sorted_results: List[Dict[str, Any]] = sorted(
                results.get("all_results", []),
                key=lambda x: x["precision"],
                reverse=True,
            )
            for result in sorted_results:
                params = result["params"]
                if self.validator.validate_parameter_values(
                    params=params,
                    param_ranges=self.hp.param_ranges
                ):
                    model_params = ModelParameters(**params)
                    self.task_manager.process_result(
                        OptimizationResult(
                            params=model_params,
                            precision=result["precision"],
                            threshold=result["threshold"],
                            tp_ratio=result["tp_ratio"],
                            epochs=result.get("epochs", self.config.max_epochs),
                            config_id=get_config_id(model_params),
                            start_time=result.get("start_time", time.time() - 3600),
                            end_time=result.get("end_time", time.time()),
                        ),
                        is_loaded=True,
                    )
                    valid_results += 1
                else:
                    self.logger.warning("Skipping invalid result entry")

            # Restore parameter sensitivity information
            if "parameter_sensitivity" in results:
                for param_name, sensitivity in results["parameter_sensitivity"].items():  # type: str, float
                    if param_name in self.hp.param_ranges:
                        self.knowledge.set_parameter_sensitivity(param_name, float(sensitivity))
                    else:
                        self.logger.warning(
                            "Unknown parameter in sensitivity data: %s", param_name
                        )

            # Update task duration statistics
            if "task_durations" in results:
                for epochs, duration in results["task_durations"].items():  # type: str, float
                    try:
                        self.task_manager.task_durations[int(epochs)] = float(duration)
                    except (ValueError, TypeError) as e:
                        self.logger.warning("Invalid duration data: %s", str(e))

            self.logger.info(
                "Loaded %d/%d valid results with best precision: %.4f",
                valid_results,
                len(results.get("all_results", [])),
                self.task_manager.best_precision,
            )

        except Exception as e:
            self.logger.warning("Failed to load previous results: %s", str(e))

    def _save_results(self, filename: str = None) -> None:
        """Save current optimization results to file."""
        filename = Path(filename) if filename else self.results_path
        all_results: List[Dict[str, Any]] = []
        for result in self.knowledge.get_best_results(None):  # None to get all results
            params_dict: Dict[str, Any] = asdict(result.params)
            config_id: str = result.config_id
            current_time: float = time.time()
            start_time = result.start_time
            end_time = result.end_time
            task_duration = end_time - start_time

            output: Dict[str, Any] = {
                "params": params_dict,
                "precision": float(result.precision),
                "threshold": float(result.threshold),
                "tp_ratio": float(result.tp_ratio),
                "epochs": result.epochs,
                "config_id": config_id,
                "timestamp": current_time,
                "start_time": start_time,
                "end_time": end_time,
                "duration": task_duration,
                "seed": self.training_params.seed,
                "init_config": asdict(self.config),
            }
            all_results.append(output)

        all_results.sort(key=lambda x: x["precision"], reverse=True)

        results: Dict[str, Any] = {
            "all_results": all_results,
            "parameter_sensitivity": self.knowledge.get_parameter_sensitivities(),
            "task_durations": self.task_manager.get_task_duration_stats(),
            "optimization_stats": self.knowledge.get_stats(),
            "total_duration": time.time() - self.start_time
        }

        if self.task_manager.best_params:
            results.update({
                "best_params": asdict(self.task_manager.best_params),
                "best_precision": float(self.task_manager.best_precision),
            })

        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            temporary = filename.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            temporary.replace(filename)
            self.logger.info("Saved optimization results to %s", filename)
        except Exception as e:
            self.logger.error("Failed to save results to %s: %s", filename, str(e))

    def _calculate_min_new_explorations(self, total_configs: int) -> int:
        """Calculate minimum number of new explorations needed.

        Args:
            total_configs: Total number of configurations evaluated so far

        Returns:
            Number of new explorations needed
        """
        if total_configs == 0:
            # If no configurations yet, need full initial points
            return self.config.num_initial_points

        if total_configs + (self.config.num_initial_points // 2) < self.config.num_initial_points:
            # If we haven't reached num_initial_points yet, explore remaining points
            return self.config.num_initial_points - total_configs

        # Default case: explore at least half of initial points
        return max(1, self.config.num_initial_points // 2)

    def _determine_next_task_type(self) -> TaskType:
        """Determine the next task type based on current state and exploration ratio."""
        total_configs: int = self.knowledge.get_result_count()
        num_initial_points: int = max(self.config.num_initial_points, len(self.gpu_indices))

        if total_configs < num_initial_points:
            return TaskType.INITIAL

        # If genetic_only is True, only use mutation tasks after initial exploration
        if self.genetic_only:
            return TaskType.MUTATION

        # Use exploration ratio to balance between exploration (mutation) and exploitation (gradient/binary)
        if self.knowledge.get_result_count() > 0:
            # Random choice weighted by exploration ratio
            if np.random.random() < self.config.exploration_ratio:
                return TaskType.MUTATION  # Exploration

            # For exploitation, alternate between gradient and binary search
            return TaskType.GRADIENT if total_configs % 2 == 0 else TaskType.BINARY

        return TaskType.MUTATION  # Default to exploration when no results

    def _submit_initial_task(self) -> bool:
        """Submit a random exploration task."""
        for _ in range(100):
            params: ModelParameters = self.sampler.sample_parameters()
            if not self.task_manager.has_been_evaluated(asdict(params)):
                self.task_manager.submit_task(params, self.config.min_epochs)
                return True
        raise RuntimeError("Unable to sample an unevaluated configuration")

    def _get_next_parameters_to_explore(self, max_count: int = None) -> List[str]:
        """Get the next parameters to explore, prioritizing least recently explored."""
        # Filter out unsuitable parameters
        valid_params: List[str] = [
            name for name, range_ in self.hp.param_ranges.items()
            if not (
                isinstance(range_, list)
                or range_.min_val == range_.max_val
                or range_.param_type == ParamType.BOOLEAN
            )
        ]

        # Sort by last exploration time (oldest first)
        sorted_params: List[str] = sorted(
            valid_params,
            key=lambda x: self.last_explored_params.get(x, 0)
        )

        if max_count:
            return sorted_params[:max_count]
        return sorted_params

    def _submit_gradient_task(self) -> bool:
        """Submit gradient exploration tasks."""
        if self.knowledge.get_result_count() > 0:
            best_results = self.knowledge.get_best_results(MAX_SENSITIVE_PARAMS)
            base_params = asdict(best_results[0].params)
        else:
            base_params = asdict(self.sampler.sample_parameters())

        # Get parameters to explore, up to number of available GPUs
        available_slots: int = min(
            self.num_gpus - self.task_manager.get_active_task_count(),
            self.config.max_configurations - self.knowledge.get_result_count(new_configs_only=True)
            - self.task_manager.get_active_task_count(),
        )
        params_to_explore: List[str] = self._get_next_parameters_to_explore(available_slots)

        tasks_submitted: int = 0
        for param_name in params_to_explore:
            # Get test points for gradient calculation based on config
            test_points: List[float] = self.param_optimizer.get_gradient_test_points(
                param_name,
                base_params[param_name],
                num_points=self.config.gradient_points_per_param
            )

            # Try to submit a task for this parameter
            for value in test_points:
                test_params: Dict[str, Any] = base_params.copy()
                test_params[param_name] = value

                if not self.task_manager.has_been_evaluated(test_params):
                    self.task_manager.submit_task(
                        ModelParameters(**test_params), self.config.min_epochs
                    )
                    self.last_explored_params[param_name] = time.time()
                    tasks_submitted += 1
                    break  # Move to next parameter after submitting one task

            if tasks_submitted >= available_slots:
                break

        return tasks_submitted > 0

    def _submit_mutation_task(self) -> bool:
        """Submit mutation tasks using genetic algorithm approach with crossover."""
        # Get promising results to use as parents
        promising_results = self.knowledge.get_promising_regions()
        if len(promising_results) < 2:  # Need at least 2 parents for crossover
            return False

        # Try to submit tasks up to available GPU slots
        available_slots = self.num_gpus - self.task_manager.get_active_task_count()
        tasks_submitted = 0

        # Create pairs of parents for crossover
        for i in range(0, len(promising_results) - 1, 2):
            if tasks_submitted >= available_slots:
                break

            parent1 = asdict(promising_results[i].params)
            parent2 = asdict(promising_results[i + 1].params)

            # First try crossover
            child_params = self.hp.crossover(parent1, parent2)

            # Then apply mutation
            child_params = self.hp.mutate(child_params, self.config.mutation_rate)

            # Convert to ModelParameters and check uniqueness
            params = ModelParameters(**child_params)
            if not self.task_manager.has_been_evaluated(child_params):
                self.task_manager.submit_task(params, self.config.max_epochs)
                tasks_submitted += 1

        return tasks_submitted > 0

    def _submit_binary_task(self) -> bool:
        """Submit binary search tasks."""
        promising_results: List[OptimizationResult] = self.knowledge.get_promising_regions()
        if not promising_results:
            return False

        base_params: Dict[str, Any] = promising_results[0].params
        if not isinstance(base_params, dict):
            base_params = asdict(base_params)

        # Get parameters sorted by sensitivity and exploration time
        self.knowledge.update_parameter_sensitivities()
        sorted_params: List[SortedParameter] = self.knowledge.get_sorted_parameters(self.last_explored_params)

        # Submit tasks up to available GPU slots
        available_slots: int = min(
            self.num_gpus - self.task_manager.get_active_task_count(),
            self.config.max_configurations - self.knowledge.get_result_count(new_configs_only=True)
            - self.task_manager.get_active_task_count(),
        )
        tasks_submitted: int = 0

        for param in sorted_params[:MAX_SENSITIVE_PARAMS]:
            param_name = param.name
            # Get test points for binary search based on config
            param_range: Any = self.hp.param_ranges[param_name]
            test_points: List[float] = self.param_optimizer.get_binary_search_points(
                param_name,
                param_range.min_val,
                param_range.max_val,
                depth=self.config.binary_search_depth
            )

            # Try to submit a task for this parameter
            for value in test_points:
                test_params: Dict[str, Any] = base_params.copy()
                test_params[param_name] = value

                if not self.task_manager.has_been_evaluated(test_params):
                    self.task_manager.submit_task(
                        ModelParameters(**test_params), self.config.max_epochs
                    )
                    self.last_explored_params[param_name] = time.time()
                    tasks_submitted += 1
                    break  # Move to next parameter after submitting one task

            if tasks_submitted >= available_slots:
                break

        return tasks_submitted > 0

    def _process_gradient_result(self, result: OptimizationResult, task: OptimizationTask) -> None:
        """Process gradient results and update parameter sensitivities."""
        if not task or task.epochs != self.config.min_epochs:
            return

        # Get base parameters from first result
        best_results = self.knowledge.get_best_results(1)
        if not best_results:
            return

        base_params: Dict[str, Any] = asdict(best_results[0].params)
        result_params: Dict[str, Any] = asdict(result.params)

        # Find explored parameter and calculate gradient
        param_name: Optional[str]
        recent_results: List[Tuple[float, float]]
        param_name, recent_results = self.knowledge.find_explored_parameter(result_params, base_params)
        if param_name and len(recent_results) >= 2:
            gradient: float = self.param_optimizer.calculate_gradient_from_results(
                param_name, recent_results
            )

            # Only process significant improvements
            best_precision = max(p for _, p in recent_results)
            base_precision = next((p for v, p in recent_results if v == base_params[param_name]), None)

            if base_precision is not None:
                improvement = best_precision - base_precision
                if abs(improvement) >= self.config.min_improvement_threshold:
                    self.knowledge.process_gradient_result(result, param_name, gradient)
                    self.logger.info(
                        "Significant improvement found for %s: %.4f (threshold: %.4f)",
                        param_name, improvement, self.config.min_improvement_threshold
                    )
                else:
                    self.logger.info(
                        "Improvement below threshold for %s: %.4f < %.4f",
                        param_name, improvement, self.config.min_improvement_threshold
                    )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "task_manager"):
            self.task_manager.cleanup()
        if hasattr(self, "worker_manager"):
            self.worker_manager.cleanup()

    def optimize(self) -> Optional[ModelParameters]:
        """Run optimization process."""
        try:
            # Initialize optimization
            if self.knowledge.get_result_count() > 0:
                self.logger.info(
                    "Continuing optimization with %d previous results. Best precision so far: %.4f",
                    self.knowledge.get_result_count(),
                    self.task_manager.best_precision,
                )
            else:
                self.logger.info("No previous results found. Starting optimization...")

            # Main optimization loop
            while self.knowledge.get_result_count(new_configs_only=True) < self.config.max_configurations:
                # Submit tasks while GPUs are available and we haven't reached max configurations
                while (
                    self.task_manager.get_active_task_count() < self.num_gpus
                    and self.knowledge.get_result_count(new_configs_only=True)
                    + self.task_manager.get_active_task_count()
                    < self.config.max_configurations
                ):
                    task_type: TaskType = self._determine_next_task_type()

                    submitted: bool = False
                    if task_type == TaskType.INITIAL:
                        submitted = self._submit_initial_task()
                    elif task_type == TaskType.GRADIENT:
                        submitted = self._submit_gradient_task()
                    elif task_type == TaskType.BINARY:
                        submitted = self._submit_binary_task()
                    else:  # TaskType.MUTATION
                        submitted = self._submit_mutation_task()

                    if not submitted:
                        self._submit_initial_task()

                # Wait for results
                try:
                    result: Optional[OptimizationResult] = self.result_queue.get(timeout=RESULT_TIMEOUT)
                    if result is None:
                        raise RuntimeError("A GPU worker failed; inspect the worker logs")

                    task: Optional[OptimizationTask] = self.task_manager.get_task_by_config_id(result.config_id)
                    if self.task_manager.process_result(result):
                        self._process_gradient_result(result, task)
                        self._save_results()

                    # Log progress
                    self.logger.info(
                        "Progress: %d/%d configs, best precision: %.4f",
                        self.knowledge.get_result_count(new_configs_only=True),
                        self.config.max_configurations,
                        self.task_manager.best_precision,
                    )
                except Exception as e:
                    self.logger.error("Error processing result: %s", str(e))
                    raise

            return self.task_manager.best_params

        finally:
            self.cleanup()
