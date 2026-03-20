"""
text_cleaner.py
---------------
Clean messy text columns before using them in a model.

What it does:
  → Converts to lowercase        "Hello World" → "hello world"
  → Removes extra spaces         "hello   world" → "hello world"
  → Removes special characters   "hello!!! :)" → "hello"
  → Optionally removes numbers
  → Optionally removes stopwords (common words like "the", "is", "at")

Usage:
    from mlforge.data_preprocessing import TextCleaner

    cleaner = TextCleaner(columns=["review", "description"])
    df      = cleaner.clean(df)

    # Remove stopwords too:
    cleaner = TextCleaner(columns=["review"], remove_stopwords=True)
"""

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "this", "that", "it", "its",
}


class TextCleaner:
    """Cleans text columns for use in ML models."""

    def __init__(self, columns: list = None,
                 lowercase:        bool = True,
                 remove_special:   bool = True,
                 remove_numbers:   bool = False,
                 remove_stopwords: bool = False):

        self.columns          = columns  # None = auto-detect all text columns
        self.lowercase        = lowercase
        self.remove_special   = remove_special
        self.remove_numbers   = remove_numbers
        self.remove_stopwords = remove_stopwords

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning to specified columns."""
        cols = (self.columns
                or df.select_dtypes(include="object").columns.tolist())

        for col in cols:
            if col not in df.columns:
                logger.warning(f"  Column '{col}' not found — skipping")
                continue
            df[col] = df[col].astype(str).apply(self._clean_text)
            df.loc[df[col].str.strip() == "", col] = None
            logger.info(f"  Cleaned text column: '{col}'")

        return df

    def _clean_text(self, text: str) -> str:
        if text in ("nan", "None", ""):
            return ""
        if self.lowercase:
            text = text.lower()
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)
        if self.remove_special:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self.remove_stopwords:
            text = " ".join(w for w in text.split() if w not in STOPWORDS)
        return text
