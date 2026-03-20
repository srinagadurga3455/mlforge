"""
tuner.py
--------
Automatically find the best hyperparameters for your model.

Usage:
    from mlforge.model_training import Tuner

    tuner   = Tuner(model_type="random_forest", task="regression")
    results = tuner.tune(df, target_col="price")

    print(results["best_params"])  # {"n_estimators": 200, "max_depth": 10}
    print(results["best_score"])   # 0.91
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

PARAM_GRIDS = {
    "random_forest"    : {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    "decision_tree"    : {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "gradient_boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    "xgboost"          : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.3], "max_depth": [3, 6]},
    "lightgbm"         : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    "ridge"            : {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "lasso"            : {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "knn"              : {"n_neighbors": [3, 5, 7, 11]},
    "svm"              : {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
}


class Tuner:
    """Finds the best hyperparameters automatically."""

    def __init__(self, model_type: str = "random_forest",
                 task: str = "regression", method: str = "random",
                 folds: int = 5, n_iter: int = 20, random_state: int = 42):
        self.model_type   = model_type
        self.task         = task
        self.method       = method
        self.folds        = folds
        self.n_iter       = n_iter
        self.random_state = random_state

    def tune(self, df: pd.DataFrame, target_col: str,
             param_grid: dict = None) -> dict:
        """Search for best hyperparameters."""
        from mlforge.model_training.trainer import Trainer, MODEL_MAP
        import importlib, inspect

        grid = param_grid or PARAM_GRIDS.get(self.model_type)
        if not grid:
            raise ValueError(f"No default grid for '{self.model_type}'.")

        X, y    = df.drop(columns=[target_col]), df[target_col]
        scoring = "r2" if self.task == "regression" else "f1_weighted"

        module_name, class_name = MODEL_MAP[self.model_type]
        module  = importlib.import_module(module_name)
        cls     = getattr(module, class_name)
        if "task" in inspect.signature(cls.__init__).parameters:
            wrapper = cls(task=self.task, random_state=self.random_state)
        else:
            wrapper = cls()
        wrapper.fit(X, y)
        sk_model = wrapper._model

        logger.info(f"Tuning '{self.model_type}' | {self.method} search | {self.folds} folds")

        if self.method == "grid":
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(sk_model, grid, cv=self.folds, scoring=scoring, n_jobs=-1)
        else:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(sk_model, grid, cv=self.folds, scoring=scoring,
                                         n_iter=self.n_iter, random_state=self.random_state, n_jobs=-1)
        search.fit(X, y)

        results = {
            "model_type" : self.model_type,
            "best_params": search.best_params_,
            "best_score" : round(float(search.best_score_), 4),
            "model"      : search.best_estimator_,
        }
        logger.info(f"Best score: {results['best_score']} | params: {results['best_params']}")
        return results
