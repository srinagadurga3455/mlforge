# data_validation — Catch data problems before they break your model

## Why does this folder exist?

**Garbage in, garbage out.**

This is the most important rule in ML. If your data has problems,
your model will learn wrong patterns — and you won't know why.

Common data problems this folder catches:

| Problem | What it means | Why it matters |
|---|---|---|
| Missing values | Some cells are empty | Model can't learn from blanks |
| Duplicate rows | Same row appears twice | Model over-learns those examples |
| Wrong data types | Age stored as "thirty" not 30 | Model sees text instead of numbers |
| Skewed columns | Most values are 0, one value is 1,000,000 | Throws off distance-based models |
| Class imbalance | 99% class A, 1% class B | Model ignores the minority class |
| Constant columns | All values are the same | Tells the model nothing |

## The ML concept: why validate before training?

Training a model takes time. Finding a data problem AFTER training
means you wasted that time and have to start over.

Validate early → fix early → train once.

## Files

| File | What it checks |
|---|---|
| `schema_validator.py` | Are the right columns present? Right types? Valid ranges? |
| `missing_values.py` | How much data is missing in each column? |
| `data_quality.py` | Duplicates, skew, imbalance, constant columns, cardinality |

## Example

```python
from mlforge.data_validation import DataQualityCheck, MissingValueCheck

# Full quality report — run this first on any new dataset
DataQualityCheck().check(df, target_col="price")

# Check for too many missing values
MissingValueCheck(threshold=0.20).check(df)
# Fails if any column has more than 20% missing values
```

## Learning ML concept: what is class imbalance?

Imagine training a fraud detector. 99% of transactions are normal,
1% are fraud. A model that always predicts "not fraud" gets 99% accuracy
— but catches zero fraud cases. That's the class imbalance problem.
DataQualityCheck detects this and warns you to use `class_weight='balanced'`.
