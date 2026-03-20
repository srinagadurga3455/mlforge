"""
feature_selector.py
-------------------
Remove features that don't help your model — or hurt it.

Methods:
    "correlation" → drop columns that are too similar to each other
                    If two columns are nearly identical, one is useless.

    "importance"  → keep only the top N most important features
                    Uses a quick Random Forest to rank features.

    "variance"    → drop columns that barely change at all
                    A column that's almost always the same value
                    gives the model nothing to learn from.

Usage:
    from mlforge.feature_engineering import FeatureSelector

    # Remove highly correlated features
    selector = FeatureSelector(method="correlation", threshold=0.95)
    df       = selector.fit_transform(df, target_col="price")

    # Keep only top 20 most important features
    selector = FeatureSelector(method="importance", top_n=20)
    df       = selector.fit_transform(df, target_col="price")

    # Apply same selection to test data:
    df_test  = selector.transform(df_test)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Removes unhelpful features."""

    def __init__(self, method: str = "correlation",
                 threshold: float = 0.95,
                 top_n:     int   = None,
                 variance_threshold: float = 0.01,
                 exclude_cols: list = None):

        self.method             = method
        self.threshold          = threshold   # for correlation
        self.top_n              = top_n       # for importance
        self.variance_threshold = variance_threshold
        self.exclude_cols       = exclude_cols or []
        self._keep_cols         = []   # set after fit_transform

    def fit_transform(self, df: pd.DataFrame,
                      target_col: str = None) -> pd.DataFrame:
        """Learn which features to keep and apply."""
        if self.method == "correlation":
            df = self._by_correlation(df, target_col)
        elif self.method == "importance":
            if not target_col:
                raise ValueError("importance method requires target_col.")
            df = self._by_importance(df, target_col)
        elif self.method == "variance":
            df = self._by_variance(df, target_col)
        else:
            raise ValueError(
                f"Unknown method: '{self.method}'. "
                f"Choose: correlation, importance, variance."
            )
        self._keep_cols = list(df.columns)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same feature selection to TEST data."""
        keep = [c for c in self._keep_cols if c in df.columns]
        return df[keep]

    def _by_correlation(self, df, target_col):
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in self.exclude_cols and c != target_col]
        corr   = df[cols].corr().abs()
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        if to_drop:
            logger.info(f"  Dropped {len(to_drop)} correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        else:
            logger.info("  No highly correlated features found.")
        return df

    def _by_importance(self, df, target_col):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in self.exclude_cols and c != target_col]
        X, y       = df[cols], df[target_col]
        is_regr    = y.dtype in ["float32", "float64"]
        model      = (RandomForestRegressor(n_estimators=50, random_state=42)
                      if is_regr
                      else RandomForestClassifier(n_estimators=50, random_state=42))
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=cols)
        n    = self.top_n or max(1, int(len(cols) * 0.7))
        keep = importances.nlargest(n).index.tolist()
        drop = [c for c in cols if c not in keep]
        if drop:
            logger.info(f"  Dropped {len(drop)} low-importance features")
        logger.info(f"  Keeping top {n} features: {keep[:5]}...")
        non_feature = [c for c in df.columns if c not in cols]
        return df[keep + non_feature]

    def _by_variance(self, df, target_col):
        cols    = [c for c in df.select_dtypes(include="number").columns
                   if c not in self.exclude_cols and c != target_col]
        low_var = [c for c in cols if df[c].var() < self.variance_threshold]
        if low_var:
            logger.info(f"  Dropped {len(low_var)} near-zero variance features: {low_var}")
            df = df.drop(columns=low_var)
        else:
            logger.info("  No near-zero variance features found.")
        return df
