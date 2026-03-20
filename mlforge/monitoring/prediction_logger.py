"""
prediction_logger.py
--------------------
Log every prediction your model makes in production.

Use this to:
  - Audit what your model predicted and when
  - Track prediction statistics over time
  - Compare predictions to actual outcomes later

Usage:
    from mlforge.monitoring import PredictionLogger

    pl = PredictionLogger(log_path="logs/")

    # Log a single prediction:
    pl.log(input_data={"age": 35, "income": 50000}, prediction=180000)

    # Log many predictions at once:
    pl.log_batch(X_test, y_pred)

    # See summary statistics:
    pl.summary()
"""

import os, json, logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs predictions to disk for auditing and monitoring."""

    def __init__(self, log_path: str = "logs/predictions/"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        today          = datetime.now().strftime("%Y%m%d")
        self.log_file  = os.path.join(log_path, f"predictions_{today}.jsonl")

    def log(self, input_data: dict, prediction,
             actual=None, metadata: dict = None):
        """Log one prediction."""
        record = {
            "timestamp" : datetime.now().isoformat(),
            "input"     : input_data,
            "prediction": float(prediction) if hasattr(prediction, "__float__") else prediction,
            "actual"    : actual,
            "metadata"  : metadata or {},
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_batch(self, X: pd.DataFrame, predictions):
        """Log multiple predictions at once."""
        for i, (_, row) in enumerate(X.iterrows()):
            self.log(input_data=row.to_dict(),
                     prediction=predictions[i] if i < len(predictions) else None)
        logger.info(f"Logged {len(X)} predictions to {self.log_file}")

    def load_logs(self, date: str = None) -> pd.DataFrame:
        """Load logged predictions into a DataFrame."""
        date     = date or datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_path, f"predictions_{date}.jsonl")
        if not os.path.exists(log_file):
            return pd.DataFrame()
        records = [json.loads(line) for line in open(log_file)]
        return pd.DataFrame(records)

    def summary(self, date: str = None) -> dict:
        """Show prediction statistics for a given day."""
        df = self.load_logs(date)
        if df.empty:
            logger.info("No predictions logged yet.")
            return {}
        preds = pd.to_numeric(df["prediction"], errors="coerce").dropna()
        result = {
            "total"  : len(df),
            "mean"   : round(float(preds.mean()), 2),
            "std"    : round(float(preds.std()),  2),
            "min"    : round(float(preds.min()),  2),
            "max"    : round(float(preds.max()),  2),
        }
        logger.info(f"Predictions today: {result['total']} | "
                    f"mean={result['mean']} | std={result['std']}")
        return result
