"""
synthetic_loader.py
-------------------
Generate fake/synthetic data for testing and prototyping.

Use this when you want to:
  - Test your pipeline before real data is available
  - Quickly try out a model idea
  - Demo the library

Usage:
    from mlforge.data_sources import SyntheticLoader

    # Classification data (predict a category)
    loader = SyntheticLoader(task="classification", rows=1000, features=10)
    df     = loader.load()

    # Regression data (predict a number)
    loader = SyntheticLoader(task="regression", rows=500, features=5)
    df     = loader.load()
"""

import pandas as pd
import numpy as np
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class SyntheticLoader(DataSource):
    """Generates synthetic data for testing."""

    def __init__(self, task: str = "classification",
                 rows: int = 1000, features: int = 10,
                 random_state: int = 42):
        super().__init__(name=f"synthetic-{task}")
        self.task         = task      # "classification" or "regression"
        self.rows         = rows      # number of rows to generate
        self.features     = features  # number of feature columns
        self.random_state = random_state

    def connect(self):
        self.is_connected = True

    def load(self) -> pd.DataFrame:
        """Generate and return synthetic data as a DataFrame."""
        from sklearn.datasets import make_classification, make_regression

        np.random.seed(self.random_state)

        if self.task == "classification":
            X, y = make_classification(
                n_samples   = self.rows,
                n_features  = self.features,
                random_state= self.random_state,
            )
            label_col = "label"

        elif self.task == "regression":
            X, y = make_regression(
                n_samples   = self.rows,
                n_features  = self.features,
                noise       = 10,
                random_state= self.random_state,
            )
            label_col = "target"

        else:
            raise ValueError(
                f"Unknown task: '{self.task}'. Use 'classification' or 'regression'."
            )

        cols = [f"feature_{i+1}" for i in range(self.features)]
        df   = pd.DataFrame(X, columns=cols)
        df[label_col] = y

        logger.info(f"Generated {self.rows:,} rows × "
                    f"{self.features} features [{self.task}]")
        return df

    def close(self):
        self.is_connected = False
