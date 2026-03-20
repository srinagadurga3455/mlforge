"""
data_quality.py
---------------
Run a full data quality check before training your model.

Checks for:
  ✓ Duplicate rows         — exact copies waste memory and skew training
  ✓ Constant columns       — columns with only one value tell the model nothing
  ✓ Skewed columns         — extreme values in one direction hurt some models
  ✓ High-cardinality cols  — text columns with too many unique values
  ✓ Class imbalance        — when one class has far fewer examples than others

Usage:
    from mlforge.data_validation import DataQualityCheck

    checker = DataQualityCheck()
    report  = checker.check(df, target_col="label")

    # See all warnings:
    for warning in report["warnings"]:
        print(warning)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataQualityCheck:
    """Runs a full data quality check and reports problems."""

    def __init__(self,
                 imbalance_threshold:   float = 0.10,
                 skew_threshold:        float = 1.0,
                 cardinality_threshold: int   = 50):

        self.imbalance_threshold   = imbalance_threshold
        # warn if minority class has less than 10% of rows

        self.skew_threshold        = skew_threshold
        # warn if column skew is greater than 1.0

        self.cardinality_threshold = cardinality_threshold
        # warn if text column has more than 50 unique values

    def check(self, df: pd.DataFrame,
              target_col: str = None) -> dict:
        """
        Run all quality checks.

        target_col → your label column — needed for class imbalance check

        Returns a report with:
            "passed"   → list of things that look good
            "warnings" → list of problems found
            "stats"    → detailed numbers
        """
        report = {"passed": [], "warnings": [], "stats": {}}

        self._check_duplicates(df, report)
        self._check_constant_columns(df, report)
        self._check_skewness(df, report)
        self._check_high_cardinality(df, report)

        if target_col and target_col in df.columns:
            self._check_class_imbalance(df, target_col, report)

        self._print_report(report)
        return report

    def _check_duplicates(self, df, report):
        n = int(df.duplicated().sum())
        report["stats"]["duplicate_rows"] = n
        if n > 0:
            pct = round(n / len(df) * 100, 1)
            report["warnings"].append(
                f"{n} duplicate rows ({pct}%). "
                f"Remove them with DataCleaner."
            )
        else:
            report["passed"].append("No duplicate rows.")

    def _check_constant_columns(self, df, report):
        cols = [c for c in df.columns if df[c].nunique() <= 1]
        report["stats"]["constant_columns"] = cols
        if cols:
            report["warnings"].append(
                f"Constant columns (useless for models): {cols}. Drop them."
            )
        else:
            report["passed"].append("No constant columns.")

    def _check_skewness(self, df, report):
        skewed = {}
        for col in df.select_dtypes(include="number").columns:
            s = df[col].skew()
            if abs(s) > self.skew_threshold:
                skewed[col] = round(float(s), 2)
        report["stats"]["skewed_columns"] = skewed
        if skewed:
            report["warnings"].append(
                f"Skewed columns: {list(skewed.keys())}. "
                f"Use Scaler(method='log') to fix."
            )
        else:
            report["passed"].append("No highly skewed columns.")

    def _check_high_cardinality(self, df, report):
        high = {c: df[c].nunique()
                for c in df.select_dtypes(include="object").columns
                if df[c].nunique() > self.cardinality_threshold}
        report["stats"]["high_cardinality_columns"] = high
        if high:
            report["warnings"].append(
                f"High-cardinality columns: {list(high.keys())}. "
                f"Use Encoder(method='target') instead of one-hot."
            )
        else:
            report["passed"].append("No high-cardinality columns.")

    def _check_class_imbalance(self, df, target_col, report):
        dist    = (df[target_col].value_counts() / len(df)).to_dict()
        min_pct = min(dist.values())
        report["stats"]["class_distribution"] = {
            str(k): round(float(v) * 100, 1) for k, v in dist.items()
        }
        if min_pct < self.imbalance_threshold:
            minority = min(dist, key=dist.get)
            report["warnings"].append(
                f"Class imbalance! '{minority}' = "
                f"{round(min_pct*100,1)}% of data. "
                f"Use ModelTrainer(class_weight='balanced')."
            )
        else:
            report["passed"].append(
                f"Classes are balanced (min: {round(min_pct*100,1)}%)."
            )

    def _print_report(self, report):
        logger.info("─" * 40)
        logger.info(f"Data Quality Check — "
                    f"{len(report['passed'])} passed, "
                    f"{len(report['warnings'])} warnings")
        for w in report["warnings"]:
            logger.warning(f"  ⚠  {w}")
        for p in report["passed"]:
            logger.info(f"  ✓  {p}")
        logger.info("─" * 40)
