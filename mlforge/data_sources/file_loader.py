"""
file_loader.py
--------------
Load data from a local file into a pandas DataFrame.

Supported formats:
    .csv     → comma-separated values  (most common)
    .xlsx    → Excel spreadsheet
    .xls     → older Excel format
    .json    → JSON file
    .parquet → compressed columnar format (fast, used for large files)

Usage:
    from mlforge.data_sources import FileLoader

    loader = FileLoader("data/train.csv")
    df     = loader.load()

    # or using 'with' (auto-closes):
    with FileLoader("data/train.csv") as loader:
        df = loader.load()

    # pass any pandas read option as keyword argument:
    loader = FileLoader("data/sales.csv", sep=";", encoding="latin-1")
"""

import os
import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)

SUPPORTED = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class FileLoader(DataSource):
    """Loads data from a local file."""

    def __init__(self, filepath: str, **read_options):
        super().__init__(name=filepath)
        self.filepath     = filepath
        self.read_options = read_options  # extra options passed to pandas

    def connect(self):
        """Check the file exists and is a supported format."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"File not found: '{self.filepath}'\n"
                f"Check the path is correct."
            )
        ext = os.path.splitext(self.filepath)[-1].lower()
        if ext not in SUPPORTED:
            raise ValueError(
                f"Unsupported format: '{ext}'\n"
                f"Supported: {', '.join(SUPPORTED)}"
            )
        self.is_connected = True
        logger.info(f"File ready: {self.filepath}")

    def load(self) -> pd.DataFrame:
        """Read the file and return a DataFrame."""
        if not self.is_connected:
            self.connect()

        ext = os.path.splitext(self.filepath)[-1].lower()
        logger.info(f"Loading {ext} file...")

        if ext == ".csv":
            df = pd.read_csv(self.filepath, **self.read_options)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(self.filepath, **self.read_options)
        elif ext == ".json":
            df = pd.read_json(self.filepath, **self.read_options)
        elif ext == ".parquet":
            df = pd.read_parquet(self.filepath, **self.read_options)

        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        self.is_connected = False
