"""
data_splitter.py
----------------
Split your data into training, validation, and test sets.

Why three sets?
  Train      → model learns from this (70% typical)
  Validation → tune hyperparameters using this (15% typical)
  Test       → final evaluation, never touched during training (15% typical)

Usage:
    from mlforge.utils import DataSplitter

    splitter = DataSplitter(train_size=0.7, val_size=0.15, test_size=0.15)
    splits   = splitter.split(df, target_col="price")

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into train, validation, and test sets."""

    def __init__(self, train_size: float = 0.70,
                 val_size:   float = 0.15,
                 test_size:  float = 0.15,
                 random_state: int = 42):
        assert abs(train_size + val_size + test_size - 1.0) < 0.01,             "train_size + val_size + test_size must equal 1.0"
        self.train_size   = train_size
        self.val_size     = val_size
        self.test_size    = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Split into train / validation / test.

        Returns a dict:
            splits["X_train"], splits["y_train"]
            splits["X_val"],   splits["y_val"]
            splits["X_test"],  splits["y_test"]
        """
        from sklearn.model_selection import train_test_split

        X, y = df.drop(columns=[target_col]), df[target_col]

        # First split: separate out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Second split: separate train from validation
        val_ratio = self.val_size / (self.train_size + self.val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio, random_state=self.random_state
        )

        logger.info(f"Split: train={len(X_train):,} | "
                    f"val={len(X_val):,} | test={len(X_test):,}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val"  : X_val,   "y_val"  : y_val,
            "X_test" : X_test,  "y_test" : y_test,
        }
