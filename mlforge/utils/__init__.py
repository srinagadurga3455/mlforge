"""
mlforge.utils
----------------
Helpful utilities for your ML projects.

    Config       — load settings from a YAML or JSON file
    DataSplitter — split data into train / validation / test sets
    setup_logger — set up clean consistent logging
"""

from .config        import Config
from .data_splitter import DataSplitter
from .logger        import setup_logger

__all__ = ["Config", "DataSplitter", "setup_logger"]
