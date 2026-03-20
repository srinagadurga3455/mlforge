# data_ingestion — Safely load and save data

## Why does this folder exist?

You might think: "I already loaded data in data_sources — why do I need ingestion too?"

Here's the difference:

- **data_sources** = HOW to load data (from CSV, Kaggle, API...)
- **data_ingestion** = WHAT TO DO with data after loading it

Ingestion adds three important things:
1. **Saves a copy** to disk with a timestamp — so you never lose raw data
2. **Logs what happened** — rows loaded, time taken, where it was saved
3. **Handles large files safely** — loads in chunks, never crashes

## The ML concept: why save raw data?

This is called the **raw data layer** in professional ML projects.

Rule #1 in production ML: **never modify your original data**.
Always save a copy before cleaning. Why?
- If your cleaning code has a bug, you can go back to the original
- You can replay the whole pipeline later
- You have a record of exactly what data was used to train each model

## Files

| File | What it does |
|---|---|
| `batch_ingestion.py` | Load a full dataset, save it, log what happened |
| `streaming_ingestion.py` | Process real-time data chunk by chunk as it arrives |

## Example

```python
from mlforge.data_sources   import FileLoader
from mlforge.data_ingestion import BatchIngestion

# Load AND save a timestamped copy automatically
ingestion = BatchIngestion(
    loader      = FileLoader("data/train.csv"),
    output_path = "data/ingested/",
)
df = ingestion.run()
# Saved as: data/ingested/data_20240315_143022.parquet
```

## Learning ML concept: what is Parquet?

Parquet is a file format that is much faster and smaller than CSV
for large datasets. mlforge saves ingested data as Parquet automatically.
A 100MB CSV might become a 20MB Parquet file that loads 10x faster.
