"""
logger.py
---------
Set up clean, consistent logging across your project.

Usage:
    from mlforge.utils import setup_logger

    logger = setup_logger("my_project")
    logger.info("Pipeline started")

    # Also log to a file:
    logger = setup_logger("my_project", log_to_file=True, log_path="logs/")
"""

import os, logging
from datetime import datetime


def setup_logger(name: str, level=logging.INFO,
                 log_to_file: bool = False,
                 log_path: str = "logs/") -> logging.Logger:
    """
    Create a logger that prints clean, timestamped messages.

    name        → a label for this logger, e.g. "my_project"
    log_to_file → also save logs to a file
    log_path    → folder where log files are saved
    """
    logger    = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger   # already configured

    fmt     = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                  datefmt="%H:%M:%S")

    # Console handler — prints to terminal
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — saves to disk
    if log_to_file:
        os.makedirs(log_path, exist_ok=True)
        today    = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(log_path, f"{name}_{today}.log")
        fh       = logging.FileHandler(filepath)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to: {filepath}")

    return logger
