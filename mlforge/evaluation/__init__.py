"""
mlforge.evaluation
---------------------
Measure how good your model is.

    Metrics   — accuracy, F1, ROC-AUC, confusion matrix, R², MAE, RMSE
    Validator — overfit check, feature importance, prediction stats
"""

from .metrics   import Metrics
from .validator import Validator

__all__ = ["Metrics", "Validator"]
