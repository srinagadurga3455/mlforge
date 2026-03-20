"""
trainer.py
----------
Train any model with one simple call.

Usage:
    from mlforge.model_training import Trainer

    trainer = Trainer(task="regression")
    results = trainer.train(df, target_col="price",
                            model_type="random_forest")

    results["score"]    # {"r2": 0.91, "mae": 1230, "rmse": 1850}
    results["model"]    # the trained model object

    y_pred = trainer.predict(X_test)
    trainer.cross_validate(df, target_col="price", folds=5)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "random_forest"    : ("mlforge.models", "RandomForest"),
    "decision_tree"    : ("mlforge.models", "DecisionTree"),
    "gradient_boosting": ("mlforge.models", "GradientBoosting"),
    "xgboost"          : ("mlforge.models", "XGBoost"),
    "lightgbm"         : ("mlforge.models", "LightGBM"),
    "linear"           : ("mlforge.models", "LinearRegression"),
    "ridge"            : ("mlforge.models", "Ridge"),
    "lasso"            : ("mlforge.models", "Lasso"),
    "logistic"         : ("mlforge.models", "LogisticRegression"),
    "svm"              : ("mlforge.models", "SVM"),
    "svr"              : ("mlforge.models", "SVR"),
    "knn"              : ("mlforge.models", "KNN"),
}


class Trainer:
    """Trains any model with a single method call."""

    def __init__(self, task: str = "regression",
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.task         = task
        self.test_size    = test_size
        self.random_state = random_state
        self.model        = None
        self.results      = {}

    def train(self, df: pd.DataFrame, target_col: str,
              model_type: str = "random_forest", **model_kwargs) -> dict:
        """Train a model and return results dict."""
        logger.info(f"Training '{model_type}' for {self.task}...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = self._split(X, y)
        self.model  = self._build_model(model_type, **model_kwargs)
        start       = datetime.now()
        self.model.fit(X_train, y_train)
        duration    = (datetime.now() - start).seconds
        score       = self._score(X_test, y_test)
        self.results = {
            "model": self.model, "model_type": model_type, "task": self.task,
            "score": score, "features": list(X.columns), "target": target_col,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration_s": duration,
        }
        self._print_summary()
        return self.results

    def cross_validate(self, df: pd.DataFrame, target_col: str,
                       folds: int = 5) -> dict:
        """K-fold cross validation for reliable scores."""
        from sklearn.model_selection import cross_val_score
        if self.model is None:
            raise RuntimeError("Call train() before cross_validate().")
        X, y    = df.drop(columns=[target_col]), df[target_col]
        scoring = "r2" if self.task == "regression" else "f1_weighted"
        scores  = cross_val_score(self.model._model, X, y,
                                   cv=folds, scoring=scoring)
        result  = {
            "mean": round(float(scores.mean()), 4),
            "std" : round(float(scores.std()),  4),
            "all" : [round(float(s), 4) for s in scores],
        }
        logger.info(f"CV {scoring}: {result['mean']} ± {result['std']}")
        return result

    def predict(self, X) -> np.ndarray:
        if self.model is None: raise RuntimeError("Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None: raise RuntimeError("Call train() first.")
        return self.model.predict_proba(X)

    def _split(self, X, y):
        from sklearn.model_selection import train_test_split
        stratify = y if self.task == "classification" else None
        return train_test_split(X, y, test_size=self.test_size,
                                 random_state=self.random_state,
                                 stratify=stratify)

    def _build_model(self, model_type, **kwargs):
        if model_type not in MODEL_MAP:
            raise ValueError(
                f"Unknown model: '{model_type}'.\n"
                f"Available: {list(MODEL_MAP.keys())}"
            )
        module_name, class_name = MODEL_MAP[model_type]
        import importlib, inspect
        module = importlib.import_module(module_name)
        cls    = getattr(module, class_name)
        if "task" in inspect.signature(cls.__init__).parameters:
            return cls(task=self.task, random_state=self.random_state, **kwargs)
        return cls(**kwargs)

    def _score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        if self.task == "regression":
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            return {
                "r2"  : round(float(r2_score(y_test, y_pred)), 4),
                "mae" : round(float(mean_absolute_error(y_test, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            return {
                "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
                "f1"       : round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "recall"   : round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }

    def _print_summary(self):
        logger.info("─" * 40)
        logger.info(f"Done — {self.results['model_type']} [{self.task}]")
        for k, v in self.results["score"].items():
            logger.info(f"  {k}: {v}")
        logger.info("─" * 40)
