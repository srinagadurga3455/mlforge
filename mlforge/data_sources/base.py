"""
base.py
-------
The parent class that every data loader inherits from.

Think of it as a contract:
    "Every loader MUST be able to connect(), load(), and close()"

This means you can swap FileLoader for KaggleLoader and the rest
of your code stays exactly the same.

How to build your own loader:
    from mlforge.data_sources.base import DataSource

    class MyLoader(DataSource):
        def connect(self): ...
        def load(self):    ...
        def close(self):   ...
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt= "%H:%M:%S",
)


class DataSource(ABC):
    """Base class for all data loaders in mlforge."""

    def __init__(self, name: str):
        self.name         = name   # label for this loader, e.g. "train.csv"
        self.is_connected = False  # True after connect() is called

    @abstractmethod
    def connect(self):
        """Open the connection / check the file / authenticate."""
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Read data and return a pandas DataFrame."""
        pass

    @abstractmethod
    def close(self):
        """Close the connection and clean up."""
        pass

    # ── 'with' statement support ─────────────────────────────────────────
    # Allows:
    #   with FileLoader("data.csv") as loader:
    #       df = loader.load()
    # The connection auto-closes when the block ends.

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        status = "connected" if self.is_connected else "not connected"
        return f"{self.__class__.__name__}('{self.name}') [{status}]"
