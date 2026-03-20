# mlforge — package root

This folder IS the library. When someone runs `pip install mlforge`,
Python installs everything inside here.

## What's inside

| Folder | What it does | Stage in ML pipeline |
|---|---|---|
| `data_sources/` | Load data from anywhere | Step 1 |
| `data_ingestion/` | Safely store loaded data | Step 2 |
| `data_validation/` | Check data quality | Step 3 |
| `data_preprocessing/` | Clean and fix data | Step 4 |
| `feature_engineering/` | Build better features | Step 5 |
| `models/` | All ML models | Step 6 |
| `model_training/` | Train, tune, save models | Step 7 |
| `evaluation/` | Measure model performance | Step 8 |
| `monitoring/` | Watch model in production | Step 9 |
| `utils/` | Helpful tools | Throughout |

## Why this structure?

A machine learning project always follows the same journey:

```
Raw data → Clean data → Good features → Trained model → Evaluated model → Deployed model
```

Each folder in mlforge matches one step of that journey.
You can use just the parts you need, or run the whole pipeline end to end.

## The `__init__.py` file

The `__init__.py` in this folder is the "front door" of the library.
It imports everything from every sub-module so you can write:

```python
import mlforge as mf   # one import gives you everything
mf.FileLoader(...)
mf.DataCleaner(...)
mf.Trainer(...)
```

Without `__init__.py`, you'd have to import from each sub-module separately.
