"""
encoder.py
----------
Convert text/category columns into numbers.
ML models only understand numbers — not words like "Male" or "London".

Methods:
    "label"     → each category gets a simple number
                  cat=0, dog=1, bird=2

    "onehot"    → each category gets its own column
                  color="red" → red=1, blue=0, green=0
                  Best for columns with fewer than ~20 unique values.

    "frequency" → replace category with how often it appears
                  "London" appears 500 times → replace with 500

    "target"    → replace category with the average target value
                  "London" has avg house price 500,000 → replace with 500000
                  Best for columns with many unique values (cities, products).

Usage:
    from mlforge.feature_engineering import Encoder

    enc = Encoder(method="onehot")
    df  = enc.fit_transform(df)                          # on training data
    df  = enc.transform(df_test)                         # on test data

    # Target encoding needs the label column:
    enc = Encoder(method="target")
    df  = enc.fit_transform(df, target_col="price")
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Encoder:
    """Encodes categorical columns into numbers."""

    def __init__(self, method: str = "onehot"):
        self.method   = method
        self._mapping = {}  # stores encoding rules learned from training

    def fit_transform(self, df: pd.DataFrame,
                      columns: list   = None,
                      target_col: str = None) -> pd.DataFrame:
        """
        Learn encoding from training data, then apply it.
        Use this on TRAINING data only.
        """
        if self.method == "target" and not target_col:
            raise ValueError(
                "Target encoding needs target_col.\n"
                "Example: enc.fit_transform(df, target_col='price')"
            )

        cols = columns or df.select_dtypes(include="object").columns.tolist()
        if target_col in cols:
            cols.remove(target_col)

        for col in cols:
            if col not in df.columns:
                continue
            if self.method == "label":
                df = self._label(df, col)
            elif self.method == "onehot":
                df = self._onehot(df, col)
            elif self.method == "frequency":
                df = self._frequency(df, col)
            elif self.method == "target":
                df = self._target(df, col, target_col)
            else:
                raise ValueError(
                    f"Unknown method: '{self.method}'. "
                    f"Choose: label, onehot, frequency, target."
                )
            logger.info(f"  Encoded '{col}' using {self.method}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same encoding to TEST data."""
        for col, mapping in self._mapping.items():
            if col not in df.columns:
                continue
            if self.method == "label":
                df[col] = df[col].map(mapping).fillna(-1)
            elif self.method == "frequency":
                df[col] = df[col].map(mapping).fillna(0)
            elif self.method == "target":
                global_mean = mapping.get("__global_mean__", 0)
                df[col] = df[col].map(mapping).fillna(global_mean)
            elif self.method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col)
                df = df.drop(columns=[col])
                for expected in mapping:
                    if expected not in dummies.columns:
                        dummies[expected] = 0
                df = pd.concat([df, dummies[mapping]], axis=1)
        return df

    def _label(self, df, col):
        mapping = {v: i for i, v in enumerate(df[col].unique())}
        self._mapping[col] = mapping
        df[col] = df[col].map(mapping)
        return df

    def _onehot(self, df, col):
        dummies = pd.get_dummies(df[col], prefix=col)
        self._mapping[col] = list(dummies.columns)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _frequency(self, df, col):
        mapping = df[col].value_counts().to_dict()
        self._mapping[col] = mapping
        df[col] = df[col].map(mapping)
        return df

    def _target(self, df, col, target_col):
        mean    = df[target_col].mean()
        mapping = df.groupby(col)[target_col].mean().to_dict()
        mapping["__global_mean__"] = mean
        self._mapping[col] = mapping
        df[col] = df[col].map(mapping).fillna(mean)
        return df
