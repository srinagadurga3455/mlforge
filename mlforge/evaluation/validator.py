"""
validator.py
------------
Check if your model is healthy — not overfitting, not underfitting.

Also shows feature importance so you can see which columns
matter most to your model.

Usage:
    from mlforge.evaluation import Validator

    v = Validator(task="regression")
    v.validate(model, X_train, X_test, y_train, y_test)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Validator:
    """Checks model health and feature importance."""

    def __init__(self, task: str = "regression"):
        self.task = task

    def validate(self, model, X_train, X_test,
                  y_train, y_test) -> dict:
        """
        Run health checks on your trained model.

        Returns:
            overfit_check      → train score vs test score
            feature_importance → which features matter most
            prediction_stats   → distribution of predictions
        """
        results = {
            "overfit_check"     : self._overfit(model, X_train, X_test, y_train, y_test),
            "feature_importance": self._importance(model, X_train),
            "prediction_stats"  : self._pred_stats(model, X_test, y_test),
        }
        self._print(results)
        return results

    def _overfit(self, model, X_train, X_test, y_train, y_test):
        if self.task == "regression":
            from sklearn.metrics import r2_score
            train_s = float(r2_score(y_train, model.predict(X_train)))
            test_s  = float(r2_score(y_test,  model.predict(X_test)))
        else:
            from sklearn.metrics import accuracy_score
            train_s = float(accuracy_score(y_train, model.predict(X_train)))
            test_s  = float(accuracy_score(y_test,  model.predict(X_test)))
        gap   = train_s - test_s
        is_of = gap > 0.1
        return {
            "train_score": round(train_s, 4),
            "test_score" : round(test_s, 4),
            "gap"        : round(gap, 4),
            "status"     : "OVERFITTING" if is_of else "HEALTHY",
        }

    def _importance(self, model, X_train):
        m = model._model if hasattr(model, "_model") else model
        if not hasattr(m, "feature_importances_"):
            return {"message": "Not available for this model type."}
        cols  = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(len(m.feature_importances_))]
        pairs = sorted(zip(cols, m.feature_importances_), key=lambda x: x[1], reverse=True)
        return {col: round(float(imp), 4) for col, imp in pairs}

    def _pred_stats(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        errors = np.array(y_test) - y_pred
        return {
            "actual_mean"   : round(float(np.mean(y_test)), 2),
            "predicted_mean": round(float(np.mean(y_pred)), 2),
            "error_mean"    : round(float(np.mean(errors)), 2),
            "error_std"     : round(float(np.std(errors)),  2),
        }

    def _print(self, results):
        oc = results["overfit_check"]
        logger.info("─" * 40)
        logger.info(f"Model Health: {oc['status']}")
        logger.info(f"  Train score : {oc['train_score']}")
        logger.info(f"  Test score  : {oc['test_score']}")
        logger.info(f"  Gap         : {oc['gap']}")
        if "feature_importances_" not in str(results["feature_importance"]):
            top5 = list(results["feature_importance"].items())[:5]
            logger.info("  Top 5 features:")
            for col, imp in top5:
                logger.info(f"    {col}: {imp}")
        logger.info("─" * 40)
