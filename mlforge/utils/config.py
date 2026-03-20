"""
config.py
---------
Load project settings from a YAML or JSON file.

Instead of hardcoding values in your code:
    FileLoader("data/train.csv")         ← hardcoded, messy
    ModelTrainer(model_type="random_forest")  ← hardcoded

Put them in config.yaml and load them here:
    config.get("data.train_path")        ← clean
    config.get("model.type")             ← clean

Usage:
    from mlforge.utils import Config

    config = Config("config.yaml")

    config.get("model.type")             # "random_forest"
    config.get("data.train_path")        # "data/train.csv"
    config.get("missing_key", "default") # "default"
"""

import os, json, logging

logger = logging.getLogger(__name__)


class Config:
    """Loads settings from a YAML or JSON file."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._settings   = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}. Using empty config.")
            return
        ext = self.config_path.split(".")[-1].lower()
        if ext in ("yaml", "yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError("Run: pip install pyyaml")
            with open(self.config_path) as f:
                self._settings = yaml.safe_load(f) or {}
        elif ext == "json":
            with open(self.config_path) as f:
                self._settings = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: .{ext}. Use .yaml or .json")
        logger.info(f"Config loaded: {self.config_path}")

    def get(self, key: str, default=None):
        """
        Get a value using dot notation.

        config.get("model.type")           # settings["model"]["type"]
        config.get("data.train_path")      # settings["data"]["train_path"]
        config.get("missing_key", "none")  # returns "none" if not found
        """
        keys  = key.split(".")
        value = self._settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def all(self) -> dict:
        """Return all settings as a dict."""
        return self._settings
