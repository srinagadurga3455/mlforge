"""
stream_loader.py
----------------
Load streaming data from Kafka or a large CSV in chunks.

Use this when data arrives in real time (live transactions,
sensor readings, etc.) or when a file is too large to load at once.

Setup:
    pip install kafka-python   (for Kafka)

Usage:
    from mlforge.data_sources import StreamLoader

    # From Kafka
    loader = StreamLoader.from_kafka(
        topic   = "transactions",
        servers = ["localhost:9092"],
    )

    # From a large CSV file in chunks
    loader = StreamLoader.from_csv("large_file.csv", chunk_size=5000)

    with loader:
        for chunk_df in loader.stream():
            process(chunk_df)   # handle each chunk
"""

import pandas as pd
import logging
from .base import DataSource

logger = logging.getLogger(__name__)


class StreamLoader(DataSource):
    """Loads streaming data chunk by chunk."""

    def __init__(self, source_type: str, **kwargs):
        super().__init__(name=f"stream:{source_type}")
        self.source_type = source_type
        self.kwargs      = kwargs
        self._consumer   = None

    @classmethod
    def from_kafka(cls, topic: str, servers: list,
                   group_id: str = "mlforge"):
        """Create a Kafka stream loader."""
        return cls("kafka", topic=topic, servers=servers, group_id=group_id)

    @classmethod
    def from_csv(cls, filepath: str, chunk_size: int = 1000):
        """Create a CSV chunk loader (for large files)."""
        return cls("csv", filepath=filepath, chunk_size=chunk_size)

    def connect(self):
        if self.source_type == "kafka":
            try:
                from kafka import KafkaConsumer
            except ImportError:
                raise ImportError("Run: pip install kafka-python")
            self._consumer = __import__("kafka").KafkaConsumer(
                self.kwargs["topic"],
                bootstrap_servers = self.kwargs["servers"],
                group_id          = self.kwargs.get("group_id", "mlforge"),
                auto_offset_reset = "earliest",
            )
        self.is_connected = True
        logger.info(f"Stream connected: {self.source_type}")

    def load(self) -> pd.DataFrame:
        """Load all chunks and return combined DataFrame."""
        chunks = list(self.stream())
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def stream(self, max_chunks: int = None):
        """
        Yield DataFrames one chunk at a time.
        Use this in a for loop to process data as it arrives.
        """
        if not self.is_connected:
            self.connect()

        if self.source_type == "csv":
            filepath   = self.kwargs["filepath"]
            chunk_size = self.kwargs.get("chunk_size", 1000)
            chunks_done = 0
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                yield chunk
                chunks_done += 1
                if max_chunks and chunks_done >= max_chunks:
                    break

        elif self.source_type == "kafka":
            import json
            chunks_done = 0
            for message in self._consumer:
                data = json.loads(message.value.decode("utf-8"))
                yield pd.DataFrame([data])
                chunks_done += 1
                if max_chunks and chunks_done >= max_chunks:
                    break

    def close(self):
        if self._consumer:
            self._consumer.close()
        self.is_connected = False
