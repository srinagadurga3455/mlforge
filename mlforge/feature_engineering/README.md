# feature_engineering — Build features that make your model smarter

## Why does this folder exist?

Raw data columns are rarely the best form for a model.

Example: you have `signup_date` and `churn_date`.
The model can't learn from dates directly.
But `days_as_customer = churn_date - signup_date` is powerful.

Feature engineering is the art of creating new columns (features)
from existing ones — and removing columns that don't help.

**This is where domain knowledge meets ML.**
A human who understands the data can create features no model
could discover on its own.

## The ML concept: features are everything

The most important factor in ML model accuracy is not which model
you use — it's the quality of your features.

A simple model with great features beats a complex model with bad features
almost every time.

## Files

| File | What it does | ML concept |
|---|---|---|
| `encoder.py` | Convert text/category → numbers | Models only understand numbers |
| `scaler.py` | Log/sqrt/robust transforms | Handles skewed distributions |
| `feature_builder.py` | Create new columns from existing ones | Feature creation |
| `feature_selector.py` | Remove columns that don't help | Dimensionality reduction |

## Encoding methods explained

| Method | How it works | Best for |
|---|---|---|
| `label` | cat=0, dog=1, bird=2 | Ordinal categories (small, medium, large) |
| `onehot` | Each category gets its own column | Categories with no order (city, colour) |
| `frequency` | Replace with how often it appears | High-cardinality columns |
| `target` | Replace with average target value | High-cardinality columns with clear patterns |

## Example

```python
from mlforge.feature_engineering import Encoder, FeatureBuilder, FeatureSelector

# Create useful new features
builder = FeatureBuilder()
df = builder.add_ratio(df, "price", "sqft", name="price_per_sqft")
df = builder.add_flag(df, "sqft", threshold=2000, name="is_large")
df = builder.extract_dates(df)   # extract year/month/day from datetime columns

# Encode categories as numbers
enc = Encoder(method="onehot")
df  = enc.fit_transform(df, target_col="price")

# Remove features that don't help (too similar to other features)
sel = FeatureSelector(method="correlation", threshold=0.95)
df  = sel.fit_transform(df, target_col="price")
```

## Learning ML concept: what is dimensionality?

Every column in your DataFrame is a "dimension". Too many dimensions
(hundreds of columns) makes models slow and often less accurate.
This is called the **curse of dimensionality**.
FeatureSelector helps by removing columns the model doesn't need.
