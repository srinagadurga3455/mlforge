"""
mongodb_loader.py
-----------------
Load data from a MongoDB collection into a pandas DataFrame.

Setup:
    pip install pymongo

Usage:
    from mlforge.data_sources import MongoDBLoader

    loader = MongoDBLoader(
        uri        = "mongodb://localhost:27017",
        database   = "my_database",
        collection = "transactions",
    )

    with loader:
        df = loader.load(query={"status": "active"}, limit=10000)
"""

import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class MongoDBLoader(DataSource):
    """Loads data from a MongoDB collection."""

    def __init__(self, uri: str, database: str, collection: str):
        super().__init__(name=f"{database}/{collection}")
        self.uri        = uri
        self.database   = database
        self.collection = collection
        self._client    = None
        self._col       = None

    def connect(self):
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("Run: pip install pymongo")

        self._client      = __import__("pymongo").MongoClient(self.uri)
        self._col         = self._client[self.database][self.collection]
        self.is_connected = True
        logger.info(f"Connected to MongoDB: {self.database}/{self.collection}")

    def load(self, query: dict = None, limit: int = None) -> pd.DataFrame:
        """
        Load documents from MongoDB into a DataFrame.

        query → MongoDB filter, e.g. {"status": "active"}
                None = load everything
        limit → max number of documents to load
        """
        if not self.is_connected:
            self.connect()

        cursor = self._col.find(query or {})
        if limit:
            cursor = cursor.limit(limit)

        records = list(cursor)
        if not records:
            logger.warning("No documents found.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])  # remove MongoDB internal ID

        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        if self._client:
            self._client.close()
        self.is_connected = False
