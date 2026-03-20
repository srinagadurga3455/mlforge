"""
mlforge.feature_engineering
-------------------------------
Build and select features that make your model smarter.

    Encoder         — convert text/category columns to numbers
    Scaler          — advanced scaling (log, robust, sqrt)
    FeatureBuilder  — create new columns from existing ones
    FeatureSelector — remove columns that don't help the model
"""

from .encoder          import Encoder
from .scaler           import Scaler
from .feature_builder  import FeatureBuilder
from .feature_selector import FeatureSelector

__all__ = ["Encoder", "Scaler", "FeatureBuilder", "FeatureSelector"]
