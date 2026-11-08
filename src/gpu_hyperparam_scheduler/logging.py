import logging
import os
import sys
from datetime import datetime

from .constants import LOGGER_FORMAT, TIME_FORMAT

__all__ = [
    'set_base_config',
    'setup_logging',
    'get_logger',
    'DEBUG',
    'INFO',
    'WARNING',
    'ERROR',
    'CRITICAL',
    'LEVEL_NAMES'  # Export custom level names
]

# Export logging levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Custom log level names (all 4 characters)
LEVEL_NAMES = {
    DEBUG: 'DBUG',
    INFO: 'INFO',
    WARNING: 'WARN',
    ERROR: 'ERRO',
    CRITICAL: 'CRIT'
}


def set_custom_logging_levels():
    for k, v in LEVEL_NAMES.items():
        # Override default level names with custom labels
        logging.addLevelName(k, v)


def set_base_config():
    logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT, datefmt=TIME_FORMAT)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration that will be used by all loggers."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, datefmt=TIME_FORMAT))

    log_filename = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, datefmt=TIME_FORMAT))

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True  # Override any existing configuration
    )


def get_logger(module: str, level: int = None) -> logging.Logger:
    """Get a logger instance. Call setup_logging() first in your main module."""
    logger = logging.getLogger(module)
    if level is not None:
        logger.setLevel(level)
    return logger


set_custom_logging_levels()

# Example usage in main module:
# from .logging import setup_logging, get_logger
# setup_logging(level=logging.INFO)  # Call this once in main module
# logger = get_logger(__name__)      # Use this in any module
