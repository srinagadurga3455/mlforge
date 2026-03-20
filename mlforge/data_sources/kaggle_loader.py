"""
kaggle_loader.py
----------------
Download and load datasets directly from Kaggle.

Setup (one time only):
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
    3. pip install kaggle

Usage:
    from mlforge.data_sources import KaggleLoader

    # Competition dataset (e.g. titanic, house-prices)
    loader = KaggleLoader(competition="titanic", save_path="data/raw/")
    df     = loader.load(filename="train.csv")

    # Public dataset (username/dataset-name from the Kaggle URL)
    loader = KaggleLoader(dataset="heptapod/titanic", save_path="data/raw/")
    df     = loader.load(filename="train.csv")
"""

import os
import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class KaggleLoader(DataSource):
    """Downloads and loads a Kaggle dataset."""

    def __init__(self, save_path: str,
                 dataset:     str = None,
                 competition: str = None):
        super().__init__(name=dataset or competition)
        self.save_path   = save_path    # where to save downloaded files
        self.dataset     = dataset      # "username/dataset-name"
        self.competition = competition  # "competition-slug"
        self.api         = None

        if not dataset and not competition:
            raise ValueError(
                "Provide either dataset='username/name' "
                "or competition='competition-name'."
            )

    def connect(self):
        """Authenticate with Kaggle API."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError("Run: pip install kaggle")

        self.api = KaggleApi()
        self.api.authenticate()
        os.makedirs(self.save_path, exist_ok=True)
        self.is_connected = True
        logger.info("Kaggle API authenticated.")

    def load(self, filename: str = None) -> pd.DataFrame:
        """Download and load a file from Kaggle."""
        if not self.is_connected:
            self.connect()

        self._download()

        # Find the file to load
        if filename:
            filepath = os.path.join(self.save_path, filename)
        else:
            csv_files = [f for f in os.listdir(self.save_path)
                         if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {self.save_path}.\n"
                    f"Files present: {os.listdir(self.save_path)}"
                )
            filepath = os.path.join(self.save_path, csv_files[0])
            logger.info(f"Auto-selected: {csv_files[0]}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"'{filename}' not found.\n"
                f"Files downloaded: {os.listdir(self.save_path)}"
            )

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def _download(self):
        if self.dataset:
            logger.info(f"Downloading dataset: {self.dataset}")
            self.api.dataset_download_files(
                self.dataset, path=self.save_path, unzip=True
            )
        else:
            logger.info(f"Downloading competition: {self.competition}")
            self.api.competition_download_files(
                self.competition, path=self.save_path
            )
            import zipfile
            zip_path = os.path.join(self.save_path,
                                    f"{self.competition}.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(self.save_path)
                os.remove(zip_path)

    def close(self):
        self.api          = None
        self.is_connected = False
