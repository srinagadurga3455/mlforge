"""
scaler.py
---------
Advanced scaling methods for numeric columns.

Methods:
    "log"    → log transform — great for skewed data (prices, salaries)
               Shrinks very large values closer to smaller ones.

    "sqrt"   → square root — gentler than log, good for count data

    "robust" → scales using median and IQR instead of mean
               Works well when outliers are present.

Usage:
    from mlforge.feature_engineering import Scaler

    scaler = Scaler(method="log", exclude_cols=["label"])
    df     = scaler.fit_transform(df)       # on training data
    df     = scaler.transform(df_test)      # on test data
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Scaler:
    """Applies advanced transforms to numeric columns."""

    def __init__(self, method: str = "log",
                 exclude_cols: list = None):
        self.method       = method
        self.exclude_cols = exclude_cols or []
        self._stats       = {}

    def fit_transform(self, df: pd.DataFrame,
                      columns: list = None) -> pd.DataFrame:
        """Learn scale from training data and apply it."""
        cols = columns or [
            c for c in df.select_dtypes(include="number").columns
            if c not in self.exclude_cols
        ]
        for col in cols:
            if self.method == "log":
                self._stats[col] = "log"
                df[col] = np.log1p(df[col].clip(lower=0))

            elif self.method == "sqrt":
                self._stats[col] = "sqrt"
                df[col] = np.sqrt(df[col].clip(lower=0))

            elif self.method == "robust":
                median = df[col].median()
                iqr    = df[col].quantile(0.75) - df[col].quantile(0.25)
                self._stats[col] = {"median": median, "iqr": iqr}
                df[col] = (df[col] - median) / (iqr + 1e-8)

            logger.info(f"  Scaled '{col}' using {self.method}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same scaling to TEST data."""
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            if stats == "log":
                df[col] = np.log1p(df[col].clip(lower=0))
            elif stats == "sqrt":
                df[col] = np.sqrt(df[col].clip(lower=0))
            elif isinstance(stats, dict):
                df[col] = ((df[col] - stats["median"])
                           / (stats["iqr"] + 1e-8))
        return df
