"""
mlforge
==========
End-to-end ML pipeline library — from data loading to deployment.

Install:
    pip install mlforge              # core (sklearn models + file loading)
    pip install mlforge[all]         # everything

Three ways to import — pick what feels natural:

    # 1. Top level — like numpy
    import mlforge as mlw
    mlw.FileLoader("train.csv")
    mlw.Trainer(task="regression")

    # 2. Module — like sklearn
    from mlforge.data_sources    import FileLoader
    from mlforge.model_training  import Trainer
    from mlforge.evaluation      import Metrics

    # 3. Direct
    from mlforge import FileLoader, Trainer, Metrics
"""

__version__ = "1.0.0"
__author__  = "srinagadurga3455"
__license__ = "MIT"

# data sources
from mlforge.data_sources import (
    FileLoader, KaggleLoader, HuggingFaceLoader,
    PostgresLoader, MongoDBLoader, APILoader,
    S3Loader, ParquetLoader, ImageLoader,
    StreamLoader, SyntheticLoader,
)

# data ingestion
from mlforge.data_ingestion import BatchIngestion, StreamingIngestion

# data validation
from mlforge.data_validation import (
    SchemaValidator, MissingValueCheck, DataQualityCheck,
)

# data preprocessing
from mlforge.data_preprocessing import (
    DataCleaner, TypeFixer, OutlierHandler, Normalizer, TextCleaner,
)

# feature engineering
from mlforge.feature_engineering import (
    Encoder, Scaler, FeatureBuilder, FeatureSelector,
)

# models
from mlforge.models import (
    RandomForest, DecisionTree, GradientBoosting,
    XGBoost, LightGBM,
    LinearRegression, Ridge, Lasso, LogisticRegression,
    SVM, SVR, KNN,
)

# model training
from mlforge.model_training import Trainer, Tuner, Registry

# evaluation
from mlforge.evaluation import Metrics, Validator

# monitoring
from mlforge.monitoring import DriftDetector, PredictionLogger

# utils
from mlforge.utils import Config, DataSplitter, setup_logger

__all__ = [
    # sources
    "FileLoader", "KaggleLoader", "HuggingFaceLoader",
    "PostgresLoader", "MongoDBLoader", "APILoader",
    "S3Loader", "ParquetLoader", "ImageLoader",
    "StreamLoader", "SyntheticLoader",
    # ingestion
    "BatchIngestion", "StreamingIngestion",
    # validation
    "SchemaValidator", "MissingValueCheck", "DataQualityCheck",
    # preprocessing
    "DataCleaner", "TypeFixer", "OutlierHandler", "Normalizer", "TextCleaner",
    # features
    "Encoder", "Scaler", "FeatureBuilder", "FeatureSelector",
    # models
    "RandomForest", "DecisionTree", "GradientBoosting",
    "XGBoost", "LightGBM",
    "LinearRegression", "Ridge", "Lasso", "LogisticRegression",
    "SVM", "SVR", "KNN",
    # training
    "Trainer", "Tuner", "Registry",
    # evaluation
    "Metrics", "Validator",
    # monitoring
    "DriftDetector", "PredictionLogger",
    # utils
    "Config", "DataSplitter", "setup_logger",
]
