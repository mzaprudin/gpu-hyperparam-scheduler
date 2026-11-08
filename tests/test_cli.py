import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpu_hyperparam_scheduler.args import parse_args


def test_explicit_arguments_override_config_even_when_equal_to_default(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"epochs": 3, "cpu": True}))
    args = parse_args(["--config", str(config), "--epochs", "10"])
    assert args.epochs == 10
    assert args.cpu


@pytest.mark.parametrize("content", [{"unrecognized": True}, [1, 2], {"forward": 2, "skip": 3}])
def test_invalid_config_fails_before_loading_tensorflow(tmp_path, content):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(content))
    with pytest.raises(SystemExit) as error:
        parse_args(["--config", str(config)])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["train", "optimize"])
def test_help_without_tensorflow_initialization(command):
    result = subprocess.run([
        sys.executable, "-c",
        f"import sys; from gpu_hyperparam_scheduler.cli import {command}; sys.argv=['command','--help']; {command}()",
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "TensorFlow" not in result.stderr


def test_search_reports_no_gpu_without_starting_workers():
    result = subprocess.run([
        sys.executable, "-c", "from gpu_hyperparam_scheduler.cli import optimize; raise SystemExit(optimize())",
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "No GPUs visible" in result.stderr
    assert "Traceback" not in result.stderr


def test_full_cpu_training_and_checkpoint_evaluation(tmp_path):
    from gpu_hyperparam_scheduler.demo import make_data
    import pickle

    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "demo.pkl"
    dataset.write_bytes(pickle.dumps(make_data()))
    config = json.loads((root / "examples/cpu.json").read_text())
    config.update(data=str(dataset), save_dir=str(tmp_path / "artifacts"),
                  model_config=str(root / "examples/model-small.json"))
    config_path = tmp_path / "cpu.json"
    config_path.write_text(json.dumps(config))
    # Run from outside the checkout: imports must resolve from the installed package.
    command = [sys.executable, "-c", "from gpu_hyperparam_scheduler.cli import train; raise SystemExit(train())",
               "--config", str(config_path)]
    result = subprocess.run(command, cwd=tmp_path, env=os.environ.copy(),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "demo.keras").is_file()
    assert (artifacts / "demo.png").is_file()
    report = json.loads((artifacts / "demo.json").read_text())
    assert report["results"]["total_samples"] > 0
    assert 0 <= report["results"]["precision"] <= 1
    result = subprocess.run(command + ["--test", "--model", str(artifacts / "demo.keras")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr
