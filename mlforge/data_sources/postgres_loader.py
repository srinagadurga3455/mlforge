"""
postgres_loader.py
------------------
Load data from a PostgreSQL database into a pandas DataFrame.

Setup:
    pip install psycopg2-binary sqlalchemy

Usage:
    from mlforge.data_sources import PostgresLoader

    loader = PostgresLoader(
        host     = "localhost",
        port     = 5432,
        database = "my_database",
        username = "admin",
        password = "secret",
    )

    with loader:
        df = loader.load("SELECT * FROM customers WHERE active = true")
"""

import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class PostgresLoader(DataSource):
    """Loads data from a PostgreSQL database."""

    def __init__(self, host: str, database: str,
                 username: str, password: str, port: int = 5432):
        super().__init__(name=f"{host}/{database}")
        self.host     = host
        self.port     = port
        self.database = database
        self.username = username
        self.password = password
        self._engine  = None

    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("Run: pip install psycopg2-binary sqlalchemy")

        url = (f"postgresql://{self.username}:{self.password}"
               f"@{self.host}:{self.port}/{self.database}")
        self._engine      = create_engine(url)
        self.is_connected = True
        logger.info(f"Connected to PostgreSQL: {self.host}/{self.database}")

    def load(self, query: str) -> pd.DataFrame:
        """Run a SQL query and return the results as a DataFrame."""
        if not self.is_connected:
            self.connect()
        logger.info(f"Running query...")
        df = pd.read_sql(query, self._engine)
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        if self._engine:
            self._engine.dispose()
        self.is_connected = False
        logger.info("PostgreSQL connection closed.")
