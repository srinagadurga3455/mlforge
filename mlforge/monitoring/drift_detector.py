"""
drift_detector.py
-----------------
Alert you when production data looks different from training data.

Why this matters:
  Your model was trained on data from January.
  By June, customer behaviour has changed.
  The model predictions become less reliable.
  Drift detection catches this before users complain.

Usage:
    from mlforge.monitoring import DriftDetector

    detector = DriftDetector(threshold=0.1)

    # Learn what normal data looks like:
    detector.fit(X_train)

    # Later, check if new data looks different:
    result = detector.detect(X_new)

    if result["drift_detected"]:
        print("Data has drifted! Consider retraining.")
        print("Drifted columns:", result["drifted_columns"])
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects when production data differs from training data."""

    def __init__(self, threshold: float = 0.1):
        self.threshold      = threshold
        # If any feature's mean changes by more than this % → drift alert
        # 0.1 = alert if mean changes by more than 10%
        self._training_stats = {}

    def fit(self, df: pd.DataFrame):
        """Learn the distribution of training data. Call once after training."""
        for col in df.select_dtypes(include="number").columns:
            self._training_stats[col] = {
                "mean": float(df[col].mean()),
                "std" : float(df[col].std()),
                "min" : float(df[col].min()),
                "max" : float(df[col].max()),
            }
        logger.info(f"DriftDetector fitted on {len(self._training_stats)} columns.")

    def detect(self, new_df: pd.DataFrame) -> dict:
        """
        Compare new data against training distribution.

        Returns:
            {
                "drift_detected" : True/False,
                "drifted_columns": [...],
                "stable_columns" : [...],
                "details"        : {column: {change info}}
            }
        """
        if not self._training_stats:
            raise RuntimeError("Call fit() with training data first.")

        result = {"drift_detected": False, "drifted_columns": [],
                  "stable_columns": [], "details": {}}

        for col, stats in self._training_stats.items():
            if col not in new_df.columns:
                continue
            new_mean   = float(new_df[col].mean())
            train_mean = stats["mean"]
            pct_change = abs((new_mean - train_mean) / (train_mean + 1e-8))

            result["details"][col] = {
                "train_mean" : round(train_mean, 4),
                "new_mean"   : round(new_mean, 4),
                "change_pct" : round(pct_change * 100, 2),
                "drifted"    : pct_change > self.threshold,
            }

            if pct_change > self.threshold:
                result["drifted_columns"].append(col)
            else:
                result["stable_columns"].append(col)

        if result["drifted_columns"]:
            result["drift_detected"] = True

        self._print_report(result)
        return result

    def _print_report(self, result):
        status = "DRIFT DETECTED" if result["drift_detected"] else "STABLE"
        logger.info("─" * 40)
        logger.info(f"Drift Detection: {status}")
        logger.info(f"  Stable   : {len(result['stable_columns'])} columns")
        logger.info(f"  Drifted  : {len(result['drifted_columns'])} columns")
        if result["drifted_columns"]:
            logger.warning("  Drifted columns:")
            for col in result["drifted_columns"]:
                d = result["details"][col]
                logger.warning(f"    {col}: changed {d['change_pct']}%")
            logger.warning("  Consider retraining your model.")
        logger.info("─" * 40)
