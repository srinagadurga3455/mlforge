"""
type_fixer.py
-------------
Automatically fix column data types.

Raw data often has:
  - Numbers stored as text → "42" instead of 42
  - Dates stored as text   → "2024-01-15" instead of a datetime
  - Boolean-like values    → "yes"/"no" instead of True/False

Usage:
    from mlforge.data_preprocessing import TypeFixer

    fixer = TypeFixer()
    df    = fixer.fix(df)

    # Explicitly tell it which columns are dates:
    fixer = TypeFixer(date_columns=["signup_date", "last_login"])
    df    = fixer.fix(df)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TypeFixer:
    """Automatically fixes column data types."""

    def __init__(self, date_columns: list = None,
                 numeric_columns: list = None):
        self.date_columns    = date_columns    or []
        self.numeric_columns = numeric_columns or []
        # explicitly named columns are fixed first
        # then auto-detection runs on remaining columns

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types across the DataFrame."""
        logger.info("Fixing data types...")

        df = self._fix_explicit(df)
        df = self._auto_fix_numeric(df)
        df = self._auto_fix_dates(df)

        logger.info("Type fixing done.")
        return df

    def _fix_explicit(self, df):
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info(f"  Cast '{col}' → numeric")
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"  Cast '{col}' → datetime")
        return df

    def _auto_fix_numeric(self, df):
        # Find text columns that are actually numbers stored as strings
        for col in df.select_dtypes(include="object").columns:
            converted    = pd.to_numeric(df[col], errors="coerce")
            success_rate = converted.notna().mean()
            if success_rate > 0.8:  # 80%+ converted successfully
                df[col] = converted
                logger.info(f"  Auto-fixed '{col}' → numeric "
                            f"({success_rate*100:.0f}% converted)")
        return df

    def _auto_fix_dates(self, df):
        # Find text columns that look like dates
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(10)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample)
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"  Auto-fixed '{col}' → datetime")
            except Exception:
                pass
        return df
