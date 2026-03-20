"""
parquet_loader.py
-----------------
Load Parquet files — the fast, compressed format for large datasets.

Parquet is much faster and smaller than CSV for large data.
Use it when your CSV files are > 100MB.

Usage:
    from mlforge.data_sources import ParquetLoader

    loader = ParquetLoader("data/large_dataset.parquet")
    df     = loader.load()

    # Load only specific columns (faster for wide tables):
    df = loader.load(columns=["age", "income", "label"])
"""

import pandas as pd
import logging
import os
from .base import DataSource

logger = logging.getLogger(__name__)


class ParquetLoader(DataSource):
    """Loads Parquet files."""

    def __init__(self, filepath: str):
        super().__init__(name=filepath)
        self.filepath = filepath

    def connect(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: '{self.filepath}'")
        self.is_connected = True
        logger.info(f"Parquet file ready: {self.filepath}")

    def load(self, columns: list = None) -> pd.DataFrame:
        """
        Load the parquet file.

        columns → list of column names to load, e.g. ["age", "income"]
                  None = load all columns
        """
        if not self.is_connected:
            self.connect()

        df = pd.read_parquet(self.filepath, columns=columns)
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        self.is_connected = False
