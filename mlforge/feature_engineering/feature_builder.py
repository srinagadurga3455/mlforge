"""
feature_builder.py
------------------
Create new columns from existing ones.
Better features → better model accuracy.

Usage:
    from mlforge.feature_engineering import FeatureBuilder

    builder = FeatureBuilder()

    # Create a ratio between two columns
    df = builder.add_ratio(df, "price", "sqft", name="price_per_sqft")

    # Create a binary flag (0 or 1) based on a threshold
    df = builder.add_flag(df, "sqft", threshold=2000, name="is_large_house")

    # Multiply two columns together
    df = builder.add_interaction(df, "bedrooms", "bathrooms")

    # Auto-extract year, month, day from date columns
    df = builder.extract_dates(df)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Creates new features from existing columns."""

    def add_ratio(self, df: pd.DataFrame,
                  col_a: str, col_b: str,
                  name: str = None) -> pd.DataFrame:
        """
        Divide col_a by col_b to create a new ratio column.

        Example:
            df = builder.add_ratio(df, "price", "sqft", name="price_per_sqft")
            # price=200000, sqft=1000 → price_per_sqft=200
        """
        col_name    = name or f"{col_a}_per_{col_b}"
        df[col_name] = df[col_a] / (df[col_b] + 1e-8)
        logger.info(f"  Created ratio feature: '{col_name}'")
        return df

    def add_flag(self, df: pd.DataFrame,
                 col: str, threshold: float,
                 name: str = None) -> pd.DataFrame:
        """
        Create a 0/1 column: 1 if value > threshold, else 0.

        Example:
            df = builder.add_flag(df, "sqft", threshold=2000)
            # sqft=2500 → is_high_sqft=1
            # sqft=1200 → is_high_sqft=0
        """
        col_name    = name or f"is_high_{col}"
        df[col_name] = (df[col] > threshold).astype(int)
        logger.info(f"  Created flag feature: '{col_name}'")
        return df

    def add_interaction(self, df: pd.DataFrame,
                        col_a: str, col_b: str,
                        name: str = None) -> pd.DataFrame:
        """
        Multiply two columns — captures their combined effect.

        Example:
            df = builder.add_interaction(df, "bedrooms", "bathrooms")
            # bedrooms=3, bathrooms=2 → bedrooms_x_bathrooms=6
        """
        col_name    = name or f"{col_a}_x_{col_b}"
        df[col_name] = df[col_a] * df[col_b]
        logger.info(f"  Created interaction feature: '{col_name}'")
        return df

    def extract_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find datetime columns and extract year, month, day, day_of_week.

        Example:
            date=2024-03-15 → date_year=2024, date_month=3,
                              date_day=15, date_day_of_week=4
        """
        for col in df.select_dtypes(include="datetime64").columns:
            df[f"{col}_year"]        = df[col].dt.year
            df[f"{col}_month"]       = df[col].dt.month
            df[f"{col}_day"]         = df[col].dt.day
            df[f"{col}_day_of_week"] = df[col].dt.dayofweek
            df = df.drop(columns=[col])
            logger.info(f"  Extracted date features from '{col}'")
        return df
