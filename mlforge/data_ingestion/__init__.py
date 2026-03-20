"""
mlforge.data_ingestion
-------------------------
Safely load and save data to disk.

    BatchIngestion    — load any dataset and save a timestamped copy
    StreamingIngestion — process real-time data chunk by chunk
"""

from .batch_ingestion     import BatchIngestion
from .streaming_ingestion import StreamingIngestion

__all__ = ["BatchIngestion", "StreamingIngestion"]
