"""
metrics.py
----------
Calculate all evaluation metrics for your model.

Regression metrics:
    R²    → how well the model explains variance (1.0 = perfect)
    MAE   → average prediction error in original units
    RMSE  → similar to MAE but penalises big errors more
    MAPE  → average percentage error

Classification metrics:
    Accuracy         → % of correct predictions
    Precision        → of all positive predictions, how many were right
    Recall           → of all actual positives, how many did we catch
    F1               → balance between precision and recall
    ROC-AUC          → how well model separates classes (needs probabilities)
    Confusion matrix → table showing correct vs wrong predictions

Usage:
    from mlforge.evaluation import Metrics

    calc = Metrics(task="regression")
    result = calc.calculate(y_true, y_pred)

    # For ROC-AUC, also pass probabilities:
    result = calc.calculate(y_true, y_pred, y_proba=trainer.predict_proba(X_test))
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """Calculates evaluation metrics for any ML task."""

    def __init__(self, task: str = "regression"):
        self.task    = task
        self.results = {}

    def calculate(self, y_true, y_pred, y_proba=None) -> dict:
        """
        Calculate all metrics and return them as a dict.

        y_true   → actual values/labels from your test set
        y_pred   → predicted values/labels from your model
        y_proba  → predicted probabilities (optional, for ROC-AUC)
                   get this from: trainer.predict_proba(X_test)
        """
        if self.task == "regression":
            self.results = self._regression(y_true, y_pred)
        else:
            self.results = self._classification(y_true, y_pred, y_proba)

        self._print()
        return self.results

    def _regression(self, y_true, y_pred):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        residuals      = y_true - y_pred
        mask           = y_true != 0
        mape           = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else float("nan")
        return {
            "r2"            : round(float(r2_score(y_true, y_pred)),              4),
            "mae"           : round(float(mean_absolute_error(y_true, y_pred)),   4),
            "rmse"          : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mape"          : round(mape,                                          4),
            "residual_mean" : round(float(residuals.mean()),                       4),
            "residual_std"  : round(float(residuals.std()),                        4),
        }

    def _classification(self, y_true, y_pred, y_proba):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                      f1_score, confusion_matrix, classification_report,
                                      roc_auc_score)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        is_binary      = len(np.unique(y_true)) == 2
        res = {
            "accuracy" : round(float(accuracy_score(y_true, y_pred)),                                    4),
            "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),    4),
            "f1"       : round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),        4),
            "confusion_matrix"       : confusion_matrix(y_true, y_pred).tolist(),
            "classification_report"  : classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }
        if y_proba is not None:
            try:
                prob = y_proba[:, 1] if (y_proba.ndim == 2 and is_binary) else y_proba
                auc  = roc_auc_score(y_true, prob, multi_class="ovr" if not is_binary else "raise", average="weighted")
                res["roc_auc"] = round(float(auc), 4)
            except Exception as e:
                logger.warning(f"ROC-AUC not computed: {e}")
                res["roc_auc"] = None
        unique, counts = np.unique(y_true, return_counts=True)
        res["class_distribution"] = {str(c): {"count": int(n), "pct": round(int(n)/len(y_true)*100, 1)}
                                      for c, n in zip(unique, counts)}
        return res

    def _print(self):
        logger.info("─" * 40)
        logger.info("Evaluation Results")
        skip = {"confusion_matrix", "classification_report", "class_distribution"}
        for k, v in self.results.items():
            if k not in skip:
                logger.info(f"  {k}: {v}")
        if "confusion_matrix" in self.results:
            logger.info("  Confusion matrix:")
            for row in self.results["confusion_matrix"]:
                logger.info(f"    {row}")
        if "class_distribution" in self.results:
            logger.info("  Class distribution:")
            for cls, info in self.results["class_distribution"].items():
                logger.info(f"    class {cls}: {info['count']} ({info['pct']}%)")
        logger.info("─" * 40)
