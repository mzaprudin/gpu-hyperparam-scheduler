"""Exercise the real scheduler with an inexpensive in-process evaluator."""

from dataclasses import asdict
import json
import logging
import queue
import threading

import numpy as np
import pytest

from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.create_model import get_config_id
from gpu_hyperparam_scheduler.data_generator import MasterGenerator
from gpu_hyperparam_scheduler.demo import make_data
from gpu_hyperparam_scheduler.optimizers.ouroboros import OuroborosConfig, OuroborosInitConfig, OuroborosOptimizer
from gpu_hyperparam_scheduler.optimizers.ouroboros.optimizer import TaskType
from gpu_hyperparam_scheduler.types import OptimizationResult


class FakeWorkers:
    """Replace only model evaluation, retaining queues and the real controller."""

    def __init__(self, gpu_indices, mp_ctx, worker_base_config):
        self.config = worker_base_config
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stopped.is_set():
            try:
                task = self.config.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                break
            self.config.result_queue.put(OptimizationResult(
                params=task.params, precision=0.5 + task.params.dropout / 2,
                threshold=0.6, tp_ratio=0.1, epochs=task.epochs,
                config_id=task.config_id, start_time=task.start_time,
            ))

    def cleanup(self):
        self.stopped.set()
        self.thread.join(timeout=2)


@pytest.fixture
def make_optimizer(monkeypatch, tmp_path):
    monkeypatch.setattr("gpu_hyperparam_scheduler.optimizers.ouroboros.optimizer.WorkerManager", FakeWorkers)
    frame = make_data(series=1, rows=32)["synthetic-0"]
    frame["filter"] = True
    frame["y_class"] = np.arange(len(frame)) % 2
    master = MasterGenerator([frame], ["price"], window_size=8)
    created = []

    def factory(budget=5, **kwargs):
        optimizer = OuroborosOptimizer(OuroborosInitConfig(
            training_params=TrainingParameters(save_dir=str(tmp_path)),
            train_master=master, val_master=master, test_master=master,
            gpu_indices=[0, 1],
            config=OuroborosConfig(min_epochs=1, max_epochs=2,
                                   num_initial_points=2, max_configurations=budget,
                                   exploration_ratio=0),
            **kwargs,
        ))
        created.append(optimizer)
        return optimizer
    yield factory
    for optimizer in created:
        optimizer.cleanup()
        optimizer.task_queue.close()
        optimizer.result_queue.close()


def test_scheduler_saves_and_resumes_results(make_optimizer, tmp_path):
    optimizer = make_optimizer()
    best = optimizer.optimize()
    assert best is not None
    assert optimizer.knowledge.get_result_count(new_configs_only=True) == 5
    saved = json.loads((tmp_path / "optimization_results.json").read_text())
    assert len(saved["all_results"]) == 5
    assert all(r["epochs"] in (1, 2) for r in saved["all_results"])
    assert all(r["end_time"] >= r["start_time"] for r in saved["all_results"])
    resumed = make_optimizer(budget=2)
    assert resumed.knowledge.get_result_count() == 5
    assert resumed.knowledge.get_result_count(new_configs_only=True) == 0
    resumed.optimize()
    assert resumed.knowledge.get_result_count() == 7
    assert resumed.best_precision >= optimizer.best_precision


def test_search_advances_beyond_initial_exploration(make_optimizer):
    optimizer = make_optimizer()
    assert optimizer._determine_next_task_type() is TaskType.INITIAL
    for _ in range(2):
        params = optimizer.sampler.sample_parameters()
        optimizer.task_manager.process_result(OptimizationResult(
            params=params, precision=0.7, threshold=0.6, tp_ratio=0.1,
            epochs=1, config_id=get_config_id(params), start_time=0,
        ), is_loaded=True)
    assert optimizer._determine_next_task_type() is TaskType.GRADIENT
    optimizer.genetic_only = True
    assert optimizer._determine_next_task_type() is TaskType.MUTATION


def test_active_config_is_not_submitted_twice(make_optimizer):
    optimizer = make_optimizer()
    params = optimizer.sampler.sample_parameters()
    optimizer.task_manager.submit_task(params, 1)
    assert optimizer.task_manager.has_been_evaluated(asdict(params))


def test_resume_preserves_reported_epoch_budget(make_optimizer, tmp_path):
    optimizer = make_optimizer(budget=1)
    optimizer.optimize()
    result = json.loads((tmp_path / "optimization_results.json").read_text())["all_results"][0]
    assert result["epochs"] == 1
