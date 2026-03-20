"""
huggingface_loader.py
---------------------
Load any dataset from HuggingFace Hub into a pandas DataFrame.
HuggingFace has thousands of free ML datasets ready to use.

Setup:
    pip install datasets

Usage:
    from mlforge.data_sources import HuggingFaceLoader

    loader = HuggingFaceLoader("imdb")
    df     = loader.load(split="train")

    # See available splits first:
    loader.available_splits()  →  ["train", "test", "unsupervised"]

    # Dataset with a config/subset:
    loader = HuggingFaceLoader("glue", subset="mrpc")
    df     = loader.load(split="validation")

    # Limit rows (useful for quick testing):
    df = loader.load(split="train", max_rows=1000)
"""

import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class HuggingFaceLoader(DataSource):
    """Loads datasets from HuggingFace Hub."""

    def __init__(self, dataset_name: str, subset: str = None):
        super().__init__(name=dataset_name)
        self.dataset_name = dataset_name
        self.subset       = subset   # some datasets have sub-configs
        self._dataset     = None

    def connect(self):
        """Download the dataset from HuggingFace (cached after first run)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")

        logger.info(f"Loading '{self.dataset_name}' from HuggingFace Hub...")
        from datasets import load_dataset
        self._dataset     = load_dataset(self.dataset_name, self.subset)
        self.is_connected = True
        logger.info(f"Dataset ready. Splits: {list(self._dataset.keys())}")

    def load(self, split: str = "train",
             max_rows: int = None) -> pd.DataFrame:
        """
        Load a split into a DataFrame.

        split    → "train", "test", or "validation"
        max_rows → limit rows loaded (useful for quick testing)
        """
        if not self.is_connected:
            self.connect()

        available = list(self._dataset.keys())
        if split not in available:
            raise ValueError(
                f"Split '{split}' not found.\n"
                f"Available: {available}"
            )

        data = self._dataset[split]
        if max_rows:
            data = data.select(range(min(max_rows, len(data))))

        df = data.to_pandas()
        logger.info(f"Loaded split='{split}': "
                    f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def available_splits(self) -> list:
        """Show which splits exist: train, test, validation, etc."""
        if not self.is_connected:
            self.connect()
        return list(self._dataset.keys())

    def close(self):
        self._dataset     = None
        self.is_connected = False
