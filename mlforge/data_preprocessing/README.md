# data_preprocessing — Clean and prepare your data

## Why does this folder exist?

Real-world data is messy. It comes with:
- Missing values ("NaN", empty cells)
- Wrong data types (numbers stored as text)
- Outliers (house price of £50,000,000 when average is £300,000)
- Unnormalised numbers (age: 0–120, salary: 0–500,000)

ML models cannot handle messy data. They expect:
- No missing values
- All numbers, no text
- Reasonable value ranges
- Features on similar scales

This folder fixes all of that.

## The ML concept: why does normalisation matter?

Imagine two features:
- `age` ranges from 0 to 120
- `salary` ranges from 0 to 500,000

Without normalisation, the model thinks salary is 4,000× more
important than age just because the numbers are bigger. That's wrong.

Normalisation puts everything on the same scale so the model
can compare features fairly.

## Files

| File | What it fixes | When to use it |
|---|---|---|
| `type_fixer.py` | Numbers stored as strings, dates stored as text | Always, run first |
| `cleaner.py` | Missing values, duplicate rows, empty columns | Always |
| `outlier_handler.py` | Extreme values that don't represent reality | When you have numeric data |
| `normalizer.py` | Scales all numbers to the same range | Before distance-based models (KNN, SVM) |
| `text_cleaner.py` | Messy text columns (uppercase, punctuation, stopwords) | When you have text features |

## Order matters

Always preprocess in this order:
```
1. TypeFixer    → fix types first so other steps work correctly
2. DataCleaner  → fill missing values, remove duplicates
3. OutlierHandler → handle extremes (on clean data only)
4. Normalizer   → scale numbers (always last)
```

## Example

```python
from mlforge.data_preprocessing import TypeFixer, DataCleaner, OutlierHandler, Normalizer

df = TypeFixer().fix(df)
df = DataCleaner(fill_numeric="median").clean(df)
df = OutlierHandler(method="iqr", action="clip").fit_transform(df)

normalizer = Normalizer(method="minmax", exclude_cols=["price"])
df_train   = normalizer.fit_transform(df_train)
df_test    = normalizer.transform(df_test)   # IMPORTANT: same scale as train
```

## Learning ML concept: fit vs transform

You always call `fit_transform` on **training data** and `transform` on
**test data**. Why? The normaliser learns the min/max from training data.
If you fit separately on test data, you'd use different scales — and your
model would see different numbers than it was trained on.
