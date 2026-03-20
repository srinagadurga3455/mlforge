"""
s3_loader.py
------------
Load data from AWS S3 into a pandas DataFrame.

Setup:
    pip install boto3
    Set AWS credentials (one of):
      - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
      - AWS CLI:  aws configure
      - IAM role if running on AWS

Usage:
    from mlforge.data_sources import S3Loader

    loader = S3Loader(
        bucket   = "my-data-bucket",
        key      = "datasets/train.csv",   # path inside the bucket
        region   = "us-east-1",
    )

    with loader:
        df = loader.load()
"""

import pandas as pd
import logging
import io
from .base import DataSource

logger = logging.getLogger(__name__)


class S3Loader(DataSource):
    """Loads data from AWS S3."""

    def __init__(self, bucket: str, key: str,
                 region: str = "us-east-1",
                 aws_access_key: str = None,
                 aws_secret_key: str = None):
        super().__init__(name=f"s3://{bucket}/{key}")
        self.bucket         = bucket
        self.key            = key
        self.region         = region
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self._s3            = None

    def connect(self):
        """Connect to AWS S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError("Run: pip install boto3")

        import boto3
        kwargs = {"region_name": self.region}
        if self.aws_access_key:
            kwargs["aws_access_key_id"]     = self.aws_access_key
            kwargs["aws_secret_access_key"] = self.aws_secret_key

        self._s3          = boto3.client("s3", **kwargs)
        self.is_connected = True
        logger.info(f"Connected to S3: {self.bucket}")

    def load(self) -> pd.DataFrame:
        """Download the file from S3 and return as a DataFrame."""
        if not self.is_connected:
            self.connect()

        logger.info(f"Downloading s3://{self.bucket}/{self.key}")
        obj  = self._s3.get_object(Bucket=self.bucket, Key=self.key)
        body = obj["Body"].read()
        ext  = self.key.split(".")[-1].lower()

        if ext == "csv":
            df = pd.read_csv(io.BytesIO(body))
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(body))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(body))
        else:
            raise ValueError(f"Unsupported S3 file type: .{ext}")

        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def close(self):
        self._s3          = None
        self.is_connected = False
