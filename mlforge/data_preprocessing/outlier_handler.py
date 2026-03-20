"""
outlier_handler.py
------------------
Find and handle extreme values that can hurt your model.

Example:
    Most houses cost £100k–£500k.
    One entry shows £50,000,000 → probably a data error.
    Without handling this, your model learns wrong patterns.

Methods:
    "iqr"    → standard statistical method, works for most data
    "zscore" → based on standard deviations, better for normal distributions

Actions:
    "clip"   → cap extreme values at the boundary (keeps all rows — safer)
    "remove" → delete rows with extreme values (loses data)

Usage:
    from mlforge.data_preprocessing import OutlierHandler

    handler = OutlierHandler(method="iqr", action="clip")
    df      = handler.fit_transform(df)

    # Apply same bounds to test data:
    df_test = handler.transform(df_test)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OutlierHandler:
    """Finds and handles outliers in numeric columns."""

    def __init__(self, method: str = "iqr",
                 action: str = "clip",
                 exclude_cols: list = None,
                 iqr_factor: float = 1.5,
                 zscore_threshold: float = 3.0):

        self.method           = method
        self.action           = action
        self.exclude_cols     = exclude_cols or []
        self.iqr_factor       = iqr_factor
        # 1.5 = standard rule — anything beyond 1.5×IQR is an outlier
        # 3.0 = only catch extreme outliers

        self.zscore_threshold = zscore_threshold
        # 3.0 = flag values more than 3 standard deviations from mean

        self._bounds = {}
        # stores (lower, upper) bounds per column
        # so we can apply same bounds to test data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn bounds from data and apply them. Use on TRAINING data."""
        before = len(df)
        cols   = [c for c in df.select_dtypes(include="number").columns
                  if c not in self.exclude_cols]

        for col in cols:
            df = self._handle(df, col, fit=True)

        removed = before - len(df)
        if removed:
            logger.info(f"  Removed {removed} rows with outliers")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same bounds to TEST / VALIDATION data."""
        for col, (lower, upper) in self._bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _handle(self, df, col, fit):
        if fit:
            if self.method == "iqr":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr    = q3 - q1
                lower  = q1 - self.iqr_factor * iqr
                upper  = q3 + self.iqr_factor * iqr
            else:  # zscore
                mu, sd = df[col].mean(), df[col].std()
                lower  = mu - self.zscore_threshold * sd
                upper  = mu + self.zscore_threshold * sd
            self._bounds[col] = (lower, upper)
        else:
            lower, upper = self._bounds.get(col, (None, None))
            if lower is None:
                return df

        n = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n == 0:
            return df

        logger.info(f"  '{col}': {n} outliers "
                    f"(range: {lower:.2f} → {upper:.2f})")

        if self.action == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif self.action == "remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df
