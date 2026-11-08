"""
GPU worker process implementation for parallel optimization.
"""

import os
import threading
import time
import queue
import logging
import json
import time
from dataclasses import asdict
from gpu_hyperparam_scheduler.model_trainer import ModelTrainer
from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.types import WorkerConfig, OptimizationResult
from gpu_hyperparam_scheduler.constants import LOGGER_FORMAT, TIME_FORMAT

# Constants for timeouts and health checks
TASK_TIMEOUT = 24 * 3600  # 24 hours timeout for task completion
HEALTH_CHECK_SLEEP = 1  # Sleep time between health checks (seconds)


class WorkerError(Exception):
    """Custom exception for worker process errors."""


def gpu_worker(config: WorkerConfig):
    """Worker process for GPU-based optimization."""

    def health_check_thread():
        """Thread for handling health check requests during long evaluations."""
        if config.stop_event is None:
            raise ValueError("Stop event isn't created")

        if config.health_queue is None:
            raise ValueError("Health queue isn't created")

        while not config.stop_event.is_set():
            try:
                # Non-blocking check for health request
                config.health_queue.get_nowait()
                # Respond to health check
                config.health_queue.put(time.time())
            except queue.Empty:
                pass  # No health check request
            time.sleep(HEALTH_CHECK_SLEEP)

    health_thread = None
    logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT, datefmt=TIME_FORMAT)

    try:
        # Set main worker thread name
        threading.current_thread().name = f"{os.getpid()}"

        # Configure worker-specific logger
        worker_logger = logging.getLogger(f"{__name__}.worker{os.getpid()}")

        worker_logger.info("GPU index: %d", config.gpu_idx)

        # Start health check thread
        health_thread = threading.Thread(target=health_check_thread)
        health_thread.daemon = True  # Thread will be terminated with main thread
        health_thread.start()

        # Create trainer instance for this GPU
        local_params = TrainingParameters(**asdict(config.base_params))
        local_params.gpu_idx = config.gpu_idx  # Override GPU index for this worker
        trainer = ModelTrainer(params=local_params, verbose=0)

        # Use master generators
        trainer.prepare(
            train_master=config.train_master,
            val_master=config.val_master,
            test_master=config.test_master,
        )

        while True:
            try:
                # Get next task with timeout
                task = config.task_queue.get(timeout=TASK_TIMEOUT)

                # Check for termination signal
                if task is None:
                    break

                # Set GPU index for this task
                task.gpu_idx = config.gpu_idx

                # Update epochs in parameters
                local_params.num_epochs = task.epochs

                try:
                    trainer.params.name = task.config_id
                    # Evaluate model
                    result = trainer.evaluate(task.params)

                    # Send result back
                    optimization_result = OptimizationResult(
                        params=task.params,
                        precision=result.precision if result else 0,
                        threshold=result.optimal_threshold if result else 0,
                        tp_ratio=result.true_positives_ratio if result else 0,
                        epochs=task.epochs,
                        config_id=task.config_id,
                        start_time=task.start_time,
                    )
                    config.result_queue.put(optimization_result)

                except (ValueError, RuntimeError) as e:
                    worker_logger.error(
                        "Error in model evaluation on GPU %d: %s\n"
                        "Task details:\n"
                        "  Config ID: %s\n"
                        "  Epochs: %d\n"
                        "  Parameters:\n%s",
                        config.gpu_idx, str(e),
                        task.config_id,
                        task.epochs,
                        json.dumps(asdict(task.params), indent=4)
                    )
                    # Send error result
                    error_result = OptimizationResult(
                        params=task.params,
                        precision=0,
                        threshold=0,
                        tp_ratio=0,
                        epochs=task.epochs,
                        config_id=task.config_id,
                        start_time=task.start_time,
                    )
                    config.result_queue.put(error_result)

            except queue.Empty:
                # Task timeout, continue waiting
                continue
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:  # pylint: disable=broad-except
                worker_logger.error(
                    "Critical error in GPU %d worker: %s",
                    config.gpu_idx, str(e)
                )
                break

    except Exception as e:  # pylint: disable=broad-except
        worker_logger.error("Fatal error in worker %d: %s", config.gpu_idx, str(e))
        # Try to send error notification through queue
        try:
            config.result_queue.put(None)
        except (queue.Full, ConnectionError, BrokenPipeError) as err:
            print("Could not send error notification: %s", err)
    finally:
        # Stop health check thread
        if config.stop_event is not None:
            config.stop_event.set()
        # Wait up to 5 seconds for thread to finish
        if health_thread:
            health_thread.join(timeout=5)
