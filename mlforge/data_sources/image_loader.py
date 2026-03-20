"""
image_loader.py
---------------
Load image datasets from a folder into a pandas DataFrame.

Expects this folder structure:
    images/
        cat/
            img001.jpg
            img002.jpg
        dog/
            img003.jpg

Each sub-folder name becomes the label.

Setup:
    pip install Pillow

Usage:
    from mlforge.data_sources import ImageLoader

    loader = ImageLoader(folder="images/", image_size=(224, 224))
    df     = loader.load()
    # df has columns: filepath, label, width, height
"""

import os
import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class ImageLoader(DataSource):
    """Loads an image folder dataset."""

    def __init__(self, folder: str, image_size: tuple = None):
        super().__init__(name=folder)
        self.folder     = folder
        self.image_size = image_size  # (width, height) to resize, or None

    def connect(self):
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"Folder not found: '{self.folder}'")
        self.is_connected = True
        logger.info(f"Image folder ready: {self.folder}")

    def load(self) -> pd.DataFrame:
        """
        Scan the folder and return a DataFrame with:
        - filepath : full path to the image
        - label    : the sub-folder name (class label)
        - filename : just the filename
        """
        if not self.is_connected:
            self.connect()

        records = []

        for label in os.listdir(self.folder):
            label_path = os.path.join(self.folder, label)
            if not os.path.isdir(label_path):
                continue

            for filename in os.listdir(label_path):
                ext = os.path.splitext(filename)[-1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(label_path, filename)
                records.append({
                    "filepath": filepath,
                    "label"   : label,
                    "filename": filename,
                })

        df = pd.DataFrame(records)
        logger.info(f"Found {len(df):,} images across "
                    f"{df['label'].nunique()} classes")
        return df

    def close(self):
        self.is_connected = False
