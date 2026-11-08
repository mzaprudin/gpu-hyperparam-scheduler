"""Compatibility entry point for train."""

from gpu_hyperparam_scheduler.cli import train

if __name__ == "__main__":
    raise SystemExit(train())
