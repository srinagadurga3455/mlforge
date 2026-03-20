# mlforge

**One library. Every ML step. No boilerplate.**

mlforge is a complete, beginner-friendly ML pipeline library.
It covers every step from loading your data to monitoring your model in production —
the same way numpy handles arrays and pandas handles DataFrames.

```
pip install mlforge
```

---

## The problem this solves

Every ML project repeats the same steps:

```
load data → clean it → validate it → build features → train a model
         → evaluate it → save it → monitor it in production
```

You write this from scratch every time. mlforge packages all of it
into one clean, consistent API — so you can focus on the actual ML,
not the plumbing.

---

## Installation

```bash
# Just the core (file loading + sklearn models)
pip install mlforge

# Add Kaggle and HuggingFace datasets
pip install mlforge[kaggle,huggingface]

# Add fast boosting models
pip install mlforge[xgboost,lightgbm]

# Add deployment server
pip install mlforge[deployment]

# Add database connectors
pip install mlforge[databases]

# Install everything
pip install mlforge[all]
```

---

## Quick start — full pipeline in under 40 lines

```python
import mlforge as mlw

# ── 1. Load data ──────────────────────────────────────────────────
with mlw.FileLoader("data/train.csv") as loader:
    df = loader.load()

# ── 2. Check data quality ─────────────────────────────────────────
mlw.DataQualityCheck().check(df, target_col="price")

# ── 3. Clean ──────────────────────────────────────────────────────
df = mlw.TypeFixer().fix(df)
df = mlw.DataCleaner(fill_numeric="median").clean(df)
df = mlw.OutlierHandler(method="iqr", action="clip").fit_transform(df)

# ── 4. Feature engineering ────────────────────────────────────────
df = mlw.Encoder(method="onehot").fit_transform(df, target_col="price")
df = mlw.FeatureSelector(method="correlation").fit_transform(df, target_col="price")
df = mlw.Normalizer(method="minmax", exclude_cols=["price"]).fit_transform(df)

# ── 5. Split ──────────────────────────────────────────────────────
splits  = mlw.DataSplitter().split(df, target_col="price")
X_train = splits["X_train"];  y_train = splits["y_train"]
X_test  = splits["X_test"];   y_test  = splits["y_test"]

# ── 6. Train ──────────────────────────────────────────────────────
import pandas as pd
trainer = mlw.Trainer(task="regression")
results = trainer.train(pd.concat([X_train, y_train], axis=1), target_col="price")

# ── 7. Evaluate ───────────────────────────────────────────────────
y_pred = trainer.predict(X_test)
mlw.Metrics(task="regression").calculate(y_test, y_pred)
mlw.Validator(task="regression").validate(results["model"], X_train, X_test, y_train, y_test)

# ── 8. Save ───────────────────────────────────────────────────────
registry = mlw.Registry(save_path="models/")
registry.save(results, version="v1")
registry.promote("random_forest_v1")

# ── 9. Monitor in production ──────────────────────────────────────
detector = mlw.DriftDetector()
detector.fit(X_train)
result = detector.detect(X_test)
if result["drift_detected"]:
    print("Data has drifted — consider retraining!")
```

---

## Import styles

Use whichever feels natural:

```python
# Style 1 — top level (fewest keystrokes)
import mlforge as mlw
mlw.DataCleaner()
mlw.Trainer(task="classification")

# Style 2 — module level (explicit, like sklearn)
from mlforge.data_preprocessing import DataCleaner
from mlforge.model_training      import Trainer
from mlforge.evaluation          import Metrics

# Style 3 — direct (fastest)
from mlforge import DataCleaner, Trainer, Metrics
```

---

## Module by module

### `data_sources` — load from anywhere

Every loader works the same way:
```python
with mlw.AnyLoader(...) as loader:
    df = loader.load()
```

| Loader | Loads from | Needs |
|---|---|---|
| `FileLoader` | CSV, Excel, JSON, Parquet | — |
| `KaggleLoader` | Kaggle competitions & datasets | `pip install mlforge[kaggle]` |
| `HuggingFaceLoader` | HuggingFace Hub | `pip install mlforge[huggingface]` |
| `PostgresLoader` | PostgreSQL | `pip install mlforge[databases]` |
| `MongoDBLoader` | MongoDB | `pip install mlforge[databases]` |
| `APILoader` | REST APIs | — |
| `S3Loader` | AWS S3 | `pip install mlforge[cloud]` |
| `ParquetLoader` | Parquet files | — |
| `ImageLoader` | Image folders | `pip install mlforge[images]` |
| `StreamLoader` | Kafka / large CSV chunks | `pip install mlforge[streaming]` |
| `SyntheticLoader` | Generated test data | — |

```python
# Load a CSV
loader = mlw.FileLoader("data/train.csv")
df     = loader.load()

# Load from Kaggle
loader = mlw.KaggleLoader(competition="titanic", save_path="data/")
df     = loader.load(filename="train.csv")

# Load from HuggingFace
loader = mlw.HuggingFaceLoader("imdb")
df     = loader.load(split="train", max_rows=5000)

# Generate test data instantly
loader = mlw.SyntheticLoader(task="classification", rows=1000)
df     = loader.load()
```

---

### `data_ingestion` — safely load and save

```python
# Load and automatically save a timestamped copy to disk
ingestion = mlw.BatchIngestion(
    loader      = mlw.FileLoader("data/train.csv"),
    output_path = "data/ingested/",
)
df = ingestion.run()
# Saved as: data/ingested/data_20240315_143022.parquet
```

---

### `data_validation` — catch problems early

```python
# Full quality check
mlw.DataQualityCheck().check(df, target_col="price")
# Checks: duplicates, constant columns, skewed columns,
#         high-cardinality columns, class imbalance

# Check column types and value ranges
schema = {
    "age"   : {"type": "int",   "min": 0,   "max": 120},
    "price" : {"type": "float", "min": 0},
}
mlw.SchemaValidator(schema).validate(df)

# Check for too many missing values
mlw.MissingValueCheck(threshold=0.20).check(df)
```

---

### `data_preprocessing` — clean your data

```python
# Fix wrong data types (numbers stored as strings, date strings, etc.)
df = mlw.TypeFixer().fix(df)

# Fill missing values and remove duplicates
df = mlw.DataCleaner(
    fill_numeric     = "median",  # or "mean" or "zero"
    fill_categorical = "mode",    # or "unknown"
    drop_threshold   = 0.5,       # drop column if 50%+ missing
).clean(df)

# Handle outliers
df = mlw.OutlierHandler(
    method = "iqr",    # or "zscore"
    action = "clip",   # or "remove"
).fit_transform(df)

# Scale numbers to the same range
normalizer = mlw.Normalizer(method="minmax", exclude_cols=["price"])
df_train   = normalizer.fit_transform(df_train)
df_test    = normalizer.transform(df_test)

# Clean text columns
df = mlw.TextCleaner(columns=["review"], remove_stopwords=True).clean(df)
```

---

### `feature_engineering` — build better features

```python
# Encode text/category columns as numbers
enc = mlw.Encoder(method="onehot")          # or label/frequency/target
df  = enc.fit_transform(df, target_col="price")
df_test = enc.transform(df_test)

# Advanced scaling (log, robust, sqrt)
scaler = mlw.Scaler(method="log", exclude_cols=["price"])
df     = scaler.fit_transform(df)

# Create new features manually
builder = mlw.FeatureBuilder()
df = builder.add_ratio(df, "price", "sqft", name="price_per_sqft")
df = builder.add_flag(df, "sqft", threshold=2000, name="is_large")
df = builder.add_interaction(df, "bedrooms", "bathrooms")
df = builder.extract_dates(df)  # year/month/day from datetime columns

# Remove features that don't help
selector = mlw.FeatureSelector(method="correlation", threshold=0.95)
df       = selector.fit_transform(df, target_col="price")
df_test  = selector.transform(df_test)
```

---

### `models` — every model in its own file

```python
# Import any model directly
from mlforge.models import RandomForest, XGBoost, SVM

model = RandomForest(task="classification", n_estimators=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # for classification
```

**All available models:**

| Model | Regression | Classification | Import |
|---|---|---|---|
| Random Forest | ✓ | ✓ | `RandomForest` |
| Decision Tree | ✓ | ✓ | `DecisionTree` |
| Gradient Boosting | ✓ | ✓ | `GradientBoosting` |
| XGBoost | ✓ | ✓ | `XGBoost` |
| LightGBM | ✓ | ✓ | `LightGBM` |
| KNN | ✓ | ✓ | `KNN` |
| Linear Regression | ✓ | — | `LinearRegression` |
| Ridge | ✓ | — | `Ridge` |
| Lasso | ✓ | — | `Lasso` |
| SVR | ✓ | — | `SVR` |
| Logistic Regression | — | ✓ | `LogisticRegression` |
| SVM | — | ✓ | `SVM` |

> **Don't know which to pick?** Start with `RandomForest`. It works well on most problems.

**Imbalanced data** (fraud, medical, churn):
```python
# "balanced" makes the model pay equal attention to all classes
model = mlw.RandomForest(task="classification", class_weight="balanced")
```

---

### `model_training` — train, tune, save

```python
# Train with one call
trainer = mlw.Trainer(task="regression")
results = trainer.train(df, target_col="price", model_type="random_forest")
# returns: {"score": {"r2": 0.91, "mae": 1230}, "model": ..., "features": [...]}

# Cross validate for reliable scores
trainer.cross_validate(df, target_col="price", folds=5)

# Tune hyperparameters automatically
tuner   = mlw.Tuner(model_type="xgboost", task="classification", method="random")
results = tuner.tune(df, target_col="label")
print(results["best_params"])   # {"n_estimators": 200, "max_depth": 6}
print(results["best_score"])    # 0.94

# Save, version, and manage models
registry = mlw.Registry(save_path="models/")
registry.save(results, version="v1")       # → models/random_forest_v1.pkl
registry.list_models()                     # → ["random_forest_v1", ...]
registry.compare()                         # print all models + scores
registry.promote("random_forest_v1")       # → models/production_model.pkl
model = registry.load("random_forest_v1") # load back
```

---

### `evaluation` — measure performance

```python
# Regression metrics
calc = mlw.Metrics(task="regression")
calc.calculate(y_test, y_pred)
# prints: R², MAE, RMSE, MAPE, residual mean/std

# Classification metrics (pass y_proba for ROC-AUC)
calc = mlw.Metrics(task="classification")
calc.calculate(y_test, y_pred, y_proba=trainer.predict_proba(X_test))
# prints: accuracy, precision, recall, F1, ROC-AUC,
#         confusion matrix, class distribution

# Overfit check + feature importance
mlw.Validator(task="regression").validate(
    model, X_train, X_test, y_train, y_test
)
# prints: HEALTHY or OVERFITTING, top 5 features, prediction stats
```

---

### `monitoring` — watch your model in production

```python
# Learn what "normal" data looks like
detector = mlw.DriftDetector(threshold=0.1)
detector.fit(X_train)

# Check new production data
result = detector.detect(X_new)
if result["drift_detected"]:
    print("Data has changed! Drifted columns:", result["drifted_columns"])
    print("Consider retraining your model.")

# Log every prediction
pl = mlw.PredictionLogger(log_path="logs/")
pl.log_batch(X_test, y_pred)
pl.summary()    # total predictions, mean, std, min, max
```

---

### `utils` — project helpers

```python
# Load settings from config.yaml instead of hardcoding values
config  = mlw.Config("config.yaml")
config.get("model.type")         # "random_forest"
config.get("data.train_path")    # "data/train.csv"

# Split data into train / validation / test
splitter = mlw.DataSplitter(train_size=0.70, val_size=0.15, test_size=0.15)
splits   = splitter.split(df, target_col="price")
X_train, y_train = splits["X_train"], splits["y_train"]
X_test,  y_test  = splits["X_test"],  splits["y_test"]

# Clean consistent logging
logger = mlw.setup_logger("my_project", log_to_file=True, log_path="logs/")
logger.info("Pipeline started")
```

---

## Example config.yaml

```yaml
data:
  train_path : data/train.csv
  target_col : price

preprocessing:
  fill_numeric   : median
  normalization  : minmax
  outlier_action : clip
  exclude_cols   : [price]

model:
  type         : random_forest
  task         : regression
  random_state : 42

monitoring:
  drift_threshold : 0.1
  log_path        : logs/predictions/
```

```python
import mlforge as mlw

config  = mlw.Config("config.yaml")
loader  = mlw.FileLoader(config.get("data.train_path"))
trainer = mlw.Trainer(
    task         = config.get("model.task"),
    random_state = config.get("model.random_state"),
)
```

---

## Publish to PyPI

When you're ready to share with the world:

```bash
pip install build twine

# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

After that:
```bash
pip install mlforge   # anyone can install it
```

---

## License

MIT — free to use, modify, and share.
