"""
mlforge.data_sources
-----------------------
Load data from any source into a pandas DataFrame.

All loaders follow the same pattern:
    with AnyLoader(...) as loader:
        df = loader.load()

Available loaders:
    FileLoader        — CSV, Excel, JSON, Parquet
    KaggleLoader      — Kaggle competitions and datasets
    HuggingFaceLoader — HuggingFace Hub datasets
    PostgresLoader    — PostgreSQL databases
    MongoDBLoader     — MongoDB collections
    APILoader         — REST APIs
    S3Loader          — AWS S3 buckets
    ParquetLoader     — Parquet files
    ImageLoader       — Image folders
    StreamLoader      — Kafka / large CSV chunks
    SyntheticLoader   — Generated test data
"""

from .base               import DataSource
from .file_loader        import FileLoader
from .kaggle_loader      import KaggleLoader
from .huggingface_loader import HuggingFaceLoader
from .postgres_loader    import PostgresLoader
from .mongodb_loader     import MongoDBLoader
from .api_loader         import APILoader
from .s3_loader          import S3Loader
from .parquet_loader     import ParquetLoader
from .image_loader       import ImageLoader
from .stream_loader      import StreamLoader
from .synthetic_loader   import SyntheticLoader

__all__ = [
    "DataSource",
    "FileLoader", "KaggleLoader", "HuggingFaceLoader",
    "PostgresLoader", "MongoDBLoader", "APILoader",
    "S3Loader", "ParquetLoader", "ImageLoader",
    "StreamLoader", "SyntheticLoader",
]
