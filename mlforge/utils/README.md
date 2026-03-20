# utils — Helpful tools used throughout the pipeline

## Why does this folder exist?

Some tools are useful at every stage of the pipeline — not just one.
Rather than repeating them everywhere, they live here.

## Files

| File | What it does | When to use it |
|---|---|---|
| `config.py` | Load settings from a YAML/JSON file | Always — avoid hardcoding values |
| `data_splitter.py` | Split data into train/val/test sets | Before training |
| `logger.py` | Set up clean consistent logging | Always |

## The ML concept: why use a config file?

Bad practice — hardcoded values scattered in code:
```python
df      = pd.read_csv("data/train_v3_final_FINAL.csv")   # where is this?
model   = RandomForest(n_estimators=200)                  # why 200?
trainer = Trainer(test_size=0.15)                         # why 0.15?
```

Good practice — values in `config.yaml`, code stays clean:
```yaml
data:
  train_path : data/train.csv
model:
  n_estimators : 200
  test_size    : 0.15
```

```python
config  = Config("config.yaml")
df      = pd.read_csv(config.get("data.train_path"))
trainer = Trainer(test_size=config.get("model.test_size"))
```

Now changing a setting means editing ONE file — not hunting through code.

## The ML concept: train / validation / test split

Why three sets instead of just train/test?

- **Train** (70%) → model learns from this
- **Validation** (15%) → you tune hyperparameters using this
  (if you tune on test data, you're overfitting to the test set)
- **Test** (15%) → final evaluation, touch this ONCE at the very end

```python
from mlforge.utils import DataSplitter

splits  = DataSplitter(train_size=0.70, val_size=0.15, test_size=0.15).split(df, target_col="price")
X_train = splits["X_train"]
X_val   = splits["X_val"]    # use this for tuning
X_test  = splits["X_test"]   # use this ONCE at the very end
```

## Logging

```python
from mlforge.utils import setup_logger

logger = setup_logger("my_project", log_to_file=True, log_path="logs/")
logger.info("Pipeline started")
logger.warning("Missing values found in 'age' column")
logger.error("Model training failed")
```

Good logging means you can always see what your pipeline did and when.
