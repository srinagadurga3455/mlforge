"""
normalizer.py
-------------
Scale numeric columns so they're all on the same range.

Why this matters:
    age    → values from   0 to   120
    salary → values from 0 to 500,000

Without scaling, the model thinks salary is more important just
because the numbers are bigger — which is wrong.

Methods:
    "minmax"   → scales everything to 0–1 range
    "standard" → scales to mean=0, std=1 (good for normal distributions)

Usage:
    from mlforge.data_preprocessing import Normalizer

    # Always exclude your target column!
    normalizer = Normalizer(method="minmax", exclude_cols=["price"])

    df_train = normalizer.fit_transform(df_train)  # learn + apply
    df_test  = normalizer.transform(df_test)        # apply same scale
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """Scales numeric columns to the same range."""

    def __init__(self, method: str = "minmax",
                 exclude_cols: list = None):

        self.method       = method
        # "minmax"   → all values become 0 to 1
        # "standard" → mean becomes 0, std becomes 1

        self.exclude_cols = exclude_cols or []
        # columns to skip — always exclude your target/label column
        # e.g. exclude_cols=["price", "label"]

        self._stats = {}
        # stores min/max (or mean/std) learned from training data
        # so we can apply the same scale to test data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Learn the scale from this data, then apply it.
        Use this on TRAINING data.
        """
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in self.exclude_cols]

        for col in cols:
            if self.method == "minmax":
                mn, mx = df[col].min(), df[col].max()
                self._stats[col] = {"min": mn, "max": mx}
                df[col] = (df[col] - mn) / (mx - mn + 1e-8)

            elif self.method == "standard":
                mu, sd = df[col].mean(), df[col].std()
                self._stats[col] = {"mean": mu, "std": sd}
                df[col] = (df[col] - mu) / (sd + 1e-8)

            logger.info(f"  Normalized '{col}' using {self.method}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same scale learned from training data.
        Use this on TEST / VALIDATION data.
        """
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            if self.method == "minmax":
                df[col] = ((df[col] - stats["min"])
                           / (stats["max"] - stats["min"] + 1e-8))
            elif self.method == "standard":
                df[col] = (df[col] - stats["mean"]) / (stats["std"] + 1e-8)
        return df
