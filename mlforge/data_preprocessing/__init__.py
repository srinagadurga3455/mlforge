"""
mlforge.data_preprocessing
------------------------------
Clean and prepare your data for training.

    DataCleaner    — fill missing values, remove duplicates
    TypeFixer      — fix wrong data types (numbers as strings, etc.)
    OutlierHandler — find and handle extreme values
    Normalizer     — scale numbers to the same range
    TextCleaner    — clean messy text columns
"""

from .cleaner         import DataCleaner
from .type_fixer      import TypeFixer
from .outlier_handler import OutlierHandler
from .normalizer      import Normalizer
from .text_cleaner    import TextCleaner

__all__ = [
    "DataCleaner", "TypeFixer", "OutlierHandler",
    "Normalizer", "TextCleaner",
]
