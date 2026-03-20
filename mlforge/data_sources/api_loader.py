"""
api_loader.py
-------------
Load data from any REST API into a pandas DataFrame.

Usage:
    from mlforge.data_sources import APILoader

    loader = APILoader(
        url     = "https://api.example.com/data",
        headers = {"Authorization": "Bearer YOUR_TOKEN"},
        params  = {"limit": 1000, "format": "json"},
    )

    with loader:
        df = loader.load()
"""

import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class APILoader(DataSource):
    """Loads data from a REST API endpoint."""

    def __init__(self, url: str,
                 headers: dict = None,
                 params:  dict = None):
        super().__init__(name=url)
        self.url     = url
        self.headers = headers or {}
        self.params  = params  or {}
        self._response = None

    def connect(self):
        """Test the API is reachable."""
        try:
            import requests
        except ImportError:
            raise ImportError("Run: pip install requests")
        self.is_connected = True
        logger.info(f"API ready: {self.url}")

    def load(self, data_key: str = None) -> pd.DataFrame:
        """
        Call the API and return the response as a DataFrame.

        data_key → if the JSON response is nested, pass the key
                   that holds the list of records.
                   e.g. data_key="results" for {"results": [...]}
                   None = treat the whole response as the data
        """
        if not self.is_connected:
            self.connect()

        import requests
        logger.info(f"Calling API: {self.url}")
        response = requests.get(self.url, headers=self.headers,
                                params=self.params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if data_key:
            data = data[data_key]

        df = pd.DataFrame(data)
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        self.is_connected = False
