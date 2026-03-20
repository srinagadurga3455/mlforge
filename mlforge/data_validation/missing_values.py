"""
missing_values.py
-----------------
Check how much data is missing in each column.

Missing data is one of the most common problems in ML datasets.
This check helps you find it before training.

Usage:
    from mlforge.data_validation import MissingValueCheck

    # Fail if any column has more than 20% missing values
    checker = MissingValueCheck(threshold=0.20)
    result  = checker.check(df)

    # See the full missing summary
    print(result["missing_summary"])
    # {"age": 2.5, "income": 0.0, "city": 15.3}  ← percentages
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MissingValueCheck:
    """Reports missing values across all columns."""

    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold
        # maximum allowed missing percentage
        # 0.20 = fail if more than 20% of values are missing

    def check(self, df: pd.DataFrame) -> dict:
        """
        Check every column for missing values.

        Returns:
            {
                "is_valid"       : True/False,
                "passed"         : columns within threshold,
                "failed"         : columns exceeding threshold,
                "missing_summary": {column: percent_missing},
            }
        """
        results = {
            "is_valid"       : True,
            "passed"         : [],
            "failed"         : [],
            "missing_summary": {},
        }

        for col in df.columns:
            pct = round(df[col].isna().mean() * 100, 2)
            results["missing_summary"][col] = pct

            if pct / 100 > self.threshold:
                results["failed"].append(
                    f"'{col}' — {pct}% missing "
                    f"(limit: {self.threshold*100:.0f}%)"
                )
            else:
                results["passed"].append(f"'{col}' — {pct}% missing ✓")

        if results["failed"]:
            results["is_valid"] = False

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        logger.info("─" * 40)
        logger.info("Missing Value Check")
        # Show worst columns first
        sorted_cols = sorted(
            results["missing_summary"].items(),
            key=lambda x: x[1], reverse=True
        )
        for col, pct in sorted_cols:
            flag = "✗" if pct / 100 > self.threshold else "✓"
            logger.info(f"  {flag} {col}: {pct}% missing")
        logger.info("─" * 40)
