"""
registry.py
-----------
Save, load, version, and manage your trained models.

Usage:
    from mlforge.model_training import Registry

    registry = Registry(save_path="models/")
    registry.save(results, version="v1")       # save
    model = registry.load("random_forest_v1")  # load
    registry.compare()                         # see all models
    registry.promote("random_forest_v1")       # mark as production
"""

import os, json, pickle, logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Registry:
    """Saves and manages trained models."""

    def __init__(self, save_path: str = "models/"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save(self, results: dict, version: str = None) -> str:
        """Save a trained model and its info."""
        version    = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{results['model_type']}_{version}"
        model_path = os.path.join(self.save_path, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(results["model"], f)
        info = {k: v for k, v in results.items() if k != "model"}
        info.update({"version": version, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
        with open(os.path.join(self.save_path, f"{model_name}_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved: {model_name}")
        return model_name

    def load(self, model_name: str):
        """Load a saved model."""
        path = os.path.join(self.save_path, f"{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model '{model_name}' not found. Available: {self.list_models()}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_models(self) -> list:
        return [f.replace(".pkl", "") for f in os.listdir(self.save_path)
                if f.endswith(".pkl") and "production" not in f]

    def get_info(self, model_name: str) -> dict:
        path = os.path.join(self.save_path, f"{model_name}_info.json")
        return json.load(open(path)) if os.path.exists(path) else {}

    def compare(self):
        """Print all saved models and their scores."""
        for name in self.list_models():
            info = self.get_info(name)
            logger.info(f"{name} | scores: {info.get('score')} | trained: {info.get('trained_at')}")

    def promote(self, model_name: str):
        """Mark a model as production-ready."""
        model = self.load(model_name)
        with open(os.path.join(self.save_path, "production_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Promoted '{model_name}' → production_model.pkl")
