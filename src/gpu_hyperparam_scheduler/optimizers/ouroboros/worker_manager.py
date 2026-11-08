"""
Worker Manager: Handles GPU worker processes and health monitoring.

This module manages worker processes that run on GPUs, including:
- Worker initialization and cleanup
- Health monitoring and task tracking
- Task duration statistics
"""

import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Dict, List, Set

from gpu_hyperparam_scheduler.types import TypedQueue, WorkerConfig
from gpu_hyperparam_scheduler.gpu_worker import gpu_worker
from .config import WORKER_JOIN_TIMEOUT, WORKER_TERMINATE_TIMEOUT, HEALTH_CHECK_TIMEOUT


class WorkerManager:
    """Manages GPU worker processes and monitors their health."""

    def __init__(
        self,
        gpu_indices: List[int],
        mp_ctx: mp.context.BaseContext,
        worker_base_config: WorkerConfig,
    ):
        """Initialize worker manager."""
        self.logger = logging.getLogger(__name__)
        self.gpu_indices = gpu_indices
        self.mp_ctx = mp_ctx
        self.worker_base_config = worker_base_config

        # Worker state
        self.workers: List[mp.Process] = []
        self.worker_last_seen: Dict[int, float] = {}
        self.task_start_times: Dict[str, float] = {}

        # Health check communication
        self.health_check_queues: Dict[int, TypedQueue] = {}
        self.worker_stop_events: Dict[int, mp.Event] = {}
        self.health_check_threads: Dict[int, threading.Thread] = {}
        self.dead_workers: Set[int] = set()

        # Initialize workers
        self._initialize_workers()
        self._verify_workers()

    def _initialize_workers(self):
        """Initialize worker processes for each GPU."""
        for gpu_idx in self.gpu_indices:
            # Create health check queue for this worker
            health_queue: TypedQueue = TypedQueue(self.mp_ctx)
            self.health_check_queues[gpu_idx] = health_queue

            # Create stop event for health check thread
            stop_event = self.mp_ctx.Event()
            self.worker_stop_events[gpu_idx] = stop_event

            # Create worker configuration
            config = WorkerConfig(
                gpu_idx=gpu_idx,
                health_queue=health_queue,
                stop_event=stop_event,
                task_queue=self.worker_base_config.task_queue,
                result_queue=self.worker_base_config.result_queue,
                base_params=self.worker_base_config.base_params,
                train_master=self.worker_base_config.train_master,
                val_master=self.worker_base_config.val_master,
                test_master=self.worker_base_config.test_master,
            )

            # Start worker process
            worker = self.mp_ctx.Process(
                target=gpu_worker, args=(config,), name=f"GPU:{gpu_idx}"
            )
            worker.start()
            self.workers.append(worker)
            self.worker_last_seen[worker.pid] = time.time()

            # Start health check thread for this worker
            health_thread = threading.Thread(
                target=self._worker_health_check,
                args=(gpu_idx, worker),
                name=f"HC:GPU:{gpu_idx}",
                daemon=True,
            )
            health_thread.start()
            self.health_check_threads[gpu_idx] = health_thread

    def _worker_health_check(self, gpu_idx: int, worker: mp.Process):
        """Monitor worker health and responsiveness."""
        while not self.worker_stop_events[gpu_idx].is_set():
            if not worker.is_alive():
                self.logger.error(
                    "Worker GPU:%d (PID:%d) died unexpectedly", gpu_idx, worker.pid
                )
                self.dead_workers.add(gpu_idx)
                break

            try:
                # Send health check request
                self.health_check_queues[gpu_idx].put_nowait(time.time())

                # Wait for response
                try:
                    self.health_check_queues[gpu_idx].get(timeout=HEALTH_CHECK_TIMEOUT)
                    self.worker_last_seen[worker.pid] = time.time()
                except queue.Empty:
                    self.logger.warning(
                        "Worker GPU:%d (PID:%d) not responding to health check",
                        gpu_idx,
                        worker.pid,
                    )
                    # Could implement recovery logic here if needed
            except queue.Full:
                self.logger.warning("Health check queue for GPU:%d is full", gpu_idx)

            # Sleep before next check
            time.sleep(HEALTH_CHECK_TIMEOUT)

    def _verify_workers(self):
        """Verify that all workers started successfully and are responsive."""
        if not self.workers:
            raise RuntimeError("No workers were initialized")

        # Check each worker's status
        for worker in self.workers:
            if not worker.is_alive():
                raise RuntimeError(f"Worker process {worker.pid} failed to start")

    def cleanup(self):
        """Cleanup worker processes."""
        self.logger.info("Cleaning up worker processes...")

        # Stop all health check threads
        for stop_event in self.worker_stop_events.values():
            stop_event.set()

        # Send termination signal to all workers
        for _ in self.workers:
            self.worker_base_config.task_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            self.logger.info("Waiting for worker %d to finish...", worker.pid)
            worker.join(timeout=WORKER_JOIN_TIMEOUT)
            if worker.is_alive():
                self.logger.warning(
                    "Worker %d did not finish, terminating...", worker.pid
                )
                worker.terminate()
                worker.join(timeout=WORKER_TERMINATE_TIMEOUT)
                if worker.is_alive():
                    self.logger.error("Worker %d still alive, killing...", worker.pid)
                    worker.kill()

        # Wait for health check threads to finish
        for gpu_idx, thread in self.health_check_threads.items():
            thread.join(timeout=WORKER_JOIN_TIMEOUT)
            if thread.is_alive():
                self.logger.warning(
                    "Health check thread for GPU:%d did not finish", gpu_idx
                )

        self.logger.info("Cleanup complete")
