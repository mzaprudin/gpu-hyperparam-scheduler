"""
Task Manager: Centralized task management system for the Ouroboros optimizer.

This module consolidates all task-related operations including:
- Task submission and tracking
- Result processing and storage
- Task statistics management
- Knowledge base updates
"""

import logging
import time
import threading
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Any

from gpu_hyperparam_scheduler.types import OptimizationTask, OptimizationResult, TypedQueue
from gpu_hyperparam_scheduler.create_model import get_config_id, ModelParameters
from .knowledge_nexus import KnowledgeNexus
from .config import (
    HEALTH_CHECK_INTERVAL,
    TASK_DURATION_THRESHOLD,
    DEFAULT_TASK_DURATION,
)


class TaskManager:
    """Centralized task management system."""

    def __init__(
        self,
        task_queue: TypedQueue,
        result_queue: TypedQueue,
        knowledge: KnowledgeNexus,
        gpu_indices: List[int],
    ) -> None:
        """Initialize task manager."""
        self.logger = logging.getLogger(__name__)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.knowledge = knowledge
        self.gpu_indices = gpu_indices
        self.start_time = time.time()

        # Task tracking
        self.active_tasks: Dict[str, OptimizationTask] = {}  # config_id -> task
        self.task_start_times: Dict[str, float] = {}  # config_id -> start_time
        self.task_durations: Dict[int, float] = {}  # epochs -> duration
        self.evaluated_configs: Set[str] = set()  # Track evaluated configurations

        # Best result tracking
        self.best_precision: float = float("-inf")
        self.best_params: Optional[ModelParameters] = None

        # Thread control
        self._stop_event = threading.Event()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="THC"
        )
        self._health_check_thread.start()

    def submit_task(self, params: ModelParameters, epochs: int) -> str:
        """Submit a new optimization task."""
        config_id = get_config_id(params)
        gpu_idx = self.gpu_indices[len(self.active_tasks) % len(self.gpu_indices)]

        task = OptimizationTask(
            params=params,
            epochs=epochs,
            config_id=config_id,
            gpu_idx=gpu_idx,
        )

        self.task_start_times[config_id] = time.time()
        self.active_tasks[config_id] = task
        self.task_queue.put(task)

        return config_id

    def process_result(
        self, result: OptimizationResult, is_loaded: bool = False
    ) -> bool:
        """Process a single optimization result."""
        if result is None:
            self.logger.error("Worker process failed")
            return False

        # Update tracking
        self.active_tasks.pop(result.config_id, None)
        self.task_start_times.pop(result.config_id, None)
        self.evaluated_configs.add(result.config_id)

        # Update duration statistics
        duration = result.end_time - result.start_time
        self._update_task_duration(result, duration)

        # Add to knowledge base
        self.knowledge.add_result(result, is_loaded=is_loaded)
        self.logger.info(
            "Added result to knowledge base: precision=%.4f, params=%s",
            result.precision,
            asdict(result.params),
        )

        # Update best result if better
        if result.precision > self.best_precision:
            self.best_precision = result.precision
            self.best_params = result.params
            self.logger.info("New best precision: %.4f", self.best_precision)

        return True

    def _update_task_duration(
        self, result: OptimizationResult, duration: float
    ) -> None:
        """Update task duration statistics."""
        epochs = result.epochs
        if epochs not in self.task_durations:
            self.task_durations[epochs] = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.task_durations[epochs] = (
                alpha * duration + (1 - alpha) * self.task_durations[epochs]
            )

    def _check_task_health(self, config_id: str, task_duration: float):
        """Check health of a single task."""
        # Calculate expected duration based on epochs
        task = self.get_task_by_config_id(config_id=config_id)
        avg_duration = self.task_durations.get(task.epochs, float("inf"))
        if avg_duration == float("inf"):
            smaller_epochs = [e for e in self.task_durations if e < task.epochs]
            if smaller_epochs:
                max_known_epochs = max(smaller_epochs)
                avg_duration = self.task_durations[max_known_epochs] * (
                    task.epochs / max_known_epochs
                )
            else:
                avg_duration = DEFAULT_TASK_DURATION

        # Check if task is taking too long
        if task_duration > avg_duration * TASK_DURATION_THRESHOLD:
            self.logger.warning(
                "Task %s running longer than usual: %.1f hours (expected %.1f)",
                config_id,
                task_duration / 3600,
                avg_duration / 3600,
            )

    def get_task_duration_stats(self) -> Dict[str, float]:
        """Get task duration statistics."""
        return {str(k): float(v) for k, v in self.task_durations.items()}

    def has_been_evaluated(self, params: Dict[str, Any]) -> bool:
        """Check if a parameter configuration has already been evaluated."""
        config_id = get_config_id(ModelParameters(**params))
        return config_id in self.evaluated_configs or config_id in self.active_tasks

    def get_active_task_count(self) -> int:
        """Get number of currently active tasks."""
        return len(self.active_tasks)

    def get_task_by_config_id(self, config_id: str) -> Optional[OptimizationTask]:
        """Get task by config ID."""
        return self.active_tasks.get(config_id)

    def get_start_time(self, config_id: str) -> Optional[float]:
        """Get task start time by config ID."""
        return self.task_start_times.get(config_id, 0)

    def get_estimated_duration(self, epochs: int) -> float:
        """Get estimated duration for given number of epochs."""
        return self.task_durations.get(epochs, 3600)  # Default 1 hour

    def _health_check_loop(self) -> None:
        """Continuously monitor task health in a separate thread."""
        while not self._stop_event.is_set():
            current_time = time.time()

            if self.task_start_times and len(self.task_start_times) > 0:
                # Check all active tasks
                for config_id, start_time in self.task_start_times.copy().items():
                    task_duration = current_time - start_time
                    try:
                        self._check_task_health(config_id, task_duration)
                    except Exception as e:
                        self.logger.error(
                            "Error checking task health for %s: %s", config_id, e
                        )

            # Wait for next check interval
            self._stop_event.wait(HEALTH_CHECK_INTERVAL)

    def cleanup(self) -> None:
        """Cleanup resources including the health check thread."""
        self._stop_event.set()
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
