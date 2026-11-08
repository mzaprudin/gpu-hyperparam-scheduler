"""
Shared constants used across the project.
"""


# Logging format
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_FORMAT = "%(asctime)s [%(levelname)4s] [%(name)12s:%(threadName)-12s] %(filename)20s:%(lineno)-4d -> %(message)s" # processName

# MasterGenerator keys
TRAIN_MASTER = "train_master"
VAL_MASTER = "val_master"
TEST_MASTER = "test_master"
PARAMS = "params"
MASTER_GENERATOR_KEYS = [TRAIN_MASTER, VAL_MASTER, TEST_MASTER]

# File suffixes
MASTER_GENERATOR_SUFFIX = "_mg"
