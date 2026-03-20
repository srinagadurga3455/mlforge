"""
batch_ingestion.py
------------------
Safely load data into your pipeline and save it to disk.

Why use this instead of loading directly?
  - Saves a timestamped copy so you never lose raw data
  - Logs exactly what was loaded (rows, columns, time)
  - Works with any loader from data_sources

Usage:
    from mlforge.data_sources    import FileLoader
    from mlforge.data_ingestion  import BatchIngestion

    ingestion = BatchIngestion(
        loader      = FileLoader("data/train.csv"),
        output_path = "data/ingested/",
    )
    df = ingestion.run()
"""

import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchIngestion:
    """Loads data using any loader and saves a copy to disk."""

    def __init__(self, loader, output_path: str):
        self.loader      = loader       # any loader from data_sources
        self.output_path = output_path  # folder to save ingested data
        os.makedirs(output_path, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        Load data, save it, and return the DataFrame.

        The saved file is named with a timestamp so old files
        are never overwritten: data_20240115_143022.parquet
        """
        logger.info("Starting data ingestion...")
        start = datetime.now()

        # Load using whatever loader was passed in
        with self.loader as loader:
            df = loader.load()

        # Save to disk with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"data_{timestamp}.parquet"
        filepath  = os.path.join(self.output_path, filename)
        df.to_parquet(filepath, index=False)

        duration = (datetime.now() - start).seconds
        logger.info("─" * 40)
        logger.info(f"Ingestion complete")
        logger.info(f"  Rows    : {len(df):,}")
        logger.info(f"  Columns : {len(df.columns)}")
        logger.info(f"  Saved   : {filepath}")
        logger.info(f"  Time    : {duration}s")
        logger.info("─" * 40)

        return df
