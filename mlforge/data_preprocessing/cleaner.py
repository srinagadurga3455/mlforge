"""
cleaner.py
----------
Fix the most common data problems automatically.

What it fixes:
  → Fills in missing numbers  (with mean, median, or zero)
  → Fills in missing text     (with most common value or "unknown")
  → Drops columns that are mostly empty  (> 50% missing by default)
  → Removes duplicate rows

Usage:
    from mlforge.data_preprocessing import DataCleaner

    cleaner = DataCleaner()
    df      = cleaner.clean(df)

    # Custom options:
    cleaner = DataCleaner(
        fill_numeric     = "median",   # or "mean" or "zero"
        fill_categorical = "unknown",  # or "mode"
        drop_threshold   = 0.4,        # drop if 40%+ missing
    )
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Fixes missing values, bad columns, and duplicate rows."""

    def __init__(self,
                 fill_numeric:     str   = "mean",
                 fill_categorical: str   = "mode",
                 drop_threshold:   float = 0.5):

        self.fill_numeric     = fill_numeric
        # how to fill missing numbers:
        #   "mean"   → average value of the column
        #   "median" → middle value (better when outliers exist)
        #   "zero"   → fill with 0

        self.fill_categorical = fill_categorical
        # how to fill missing text:
        #   "mode"    → most common value in the column
        #   "unknown" → fill with the string "unknown"

        self.drop_threshold   = drop_threshold
        # drop a column entirely if it has more than this % missing
        # 0.5 = drop if 50%+ of values are missing

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps and return the cleaned DataFrame."""
        logger.info(f"Cleaning started — shape: {df.shape}")

        df = self._drop_mostly_empty_columns(df)
        df = self._fill_numeric_missing(df)
        df = self._fill_categorical_missing(df)
        df = self._remove_duplicates(df)

        logger.info(f"Cleaning done — shape: {df.shape}")
        return df

    def _drop_mostly_empty_columns(self, df):
        before = df.shape[1]
        for col in list(df.columns):
            if df[col].isna().mean() > self.drop_threshold:
                pct = df[col].isna().mean() * 100
                df  = df.drop(columns=[col])
                logger.info(f"  Dropped '{col}' — {pct:.0f}% missing")
        dropped = before - df.shape[1]
        if dropped:
            logger.info(f"  Dropped {dropped} mostly-empty columns")
        return df

    def _fill_numeric_missing(self, df):
        for col in df.select_dtypes(include="number").columns:
            n = df[col].isna().sum()
            if n == 0:
                continue
            if self.fill_numeric == "mean":
                value = df[col].mean()
            elif self.fill_numeric == "median":
                value = df[col].median()
            else:
                value = 0
            df[col] = df[col].fillna(value)
            logger.info(f"  Filled '{col}' ({n} missing) with {self.fill_numeric}: {value:.2f}")
        return df

    def _fill_categorical_missing(self, df):
        for col in df.select_dtypes(include="object").columns:
            n = df[col].isna().sum()
            if n == 0:
                continue
            value = df[col].mode()[0] if self.fill_categorical == "mode" else "unknown"
            df[col] = df[col].fillna(value)
            logger.info(f"  Filled '{col}' ({n} missing) with: '{value}'")
        return df

    def _remove_duplicates(self, df):
        before = len(df)
        df     = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info(f"  Removed {removed} duplicate rows")
        return df
