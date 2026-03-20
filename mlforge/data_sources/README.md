# data_sources — Load data from anywhere

## Why does this folder exist?

Every ML project starts with data. But data lives in many different places:
- A CSV file on your computer
- A Kaggle competition dataset
- A database like PostgreSQL
- An API on the internet
- AWS S3 cloud storage
- A stream of live events from Kafka

Without mlforge, you'd write different loading code for each one.
With mlforge, every loader works **exactly the same way**:

```python
with AnyLoader(...) as loader:
    df = loader.load()
```

Swap `FileLoader` for `KaggleLoader` or `S3Loader` — your pipeline code
doesn't change at all.

## The ML concept: why clean data loading matters

Before you can train a model, you need to get data INTO Python.
This sounds simple, but it's where many projects break:
- File encoding issues
- Authentication failures
- Partial downloads
- Wrong file formats

The loaders in this folder handle all of that for you.

## Files

| File | What it loads |
|---|---|
| `base.py` | The parent class all loaders inherit from |
| `file_loader.py` | CSV, Excel, JSON, Parquet from your computer |
| `kaggle_loader.py` | Any Kaggle competition or public dataset |
| `huggingface_loader.py` | Thousands of free datasets from HuggingFace Hub |
| `postgres_loader.py` | PostgreSQL databases using SQL queries |
| `mongodb_loader.py` | MongoDB document collections |
| `api_loader.py` | Any REST API endpoint |
| `s3_loader.py` | AWS S3 cloud storage buckets |
| `parquet_loader.py` | Parquet files (fast compressed format) |
| `image_loader.py` | Folders of images with class labels |
| `stream_loader.py` | Real-time Kafka streams or large CSV chunks |
| `synthetic_loader.py` | Generated fake data for testing |

## Learning ML concept: what is a DataFrame?

All loaders return a **pandas DataFrame** — think of it as an Excel
spreadsheet in Python. Rows = examples. Columns = features.
Your entire ML pipeline works on DataFrames.

```python
from mlforge.data_sources import FileLoader

loader = FileLoader("data/train.csv")
df     = loader.load()

print(df.shape)     # (1000, 15) → 1000 rows, 15 columns
print(df.head())    # see first 5 rows
print(df.columns)   # see all column names
```
