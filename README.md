# mlforge Examples

Three complete end-to-end examples showing how to use mlforge.
Each one is fully commented — every line explains what it does and why.

## Examples

### 01 — Quickstart (5 minutes)
`python examples/01_quickstart.py`

The simplest possible pipeline. Generates data, cleans it, trains a model, makes predictions.
Start here if you're new to mlforge or ML.

### 02 — Titanic Survival Prediction (Classification)
`python examples/02_titanic.py`

Predicts who survived the Titanic disaster (yes/no).
Shows how to handle real messy data — missing values, text columns, duplicates.

Requires: `titanic.csv` in the same folder.
Download: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

### 03 — House Price Prediction (Regression)
`python examples/03_house_prices.py`

Predicts house prices from features like size, bedrooms, and location.
Shows feature engineering, cross validation, hyperparameter tuning, and drift detection.

No download needed — generates realistic data automatically.

## What you'll learn

| Concept | Where it's shown |
|---|---|
| Loading data from a file | 02_titanic.py |
| Checking data quality | 02_titanic.py, 03_house_prices.py |
| Filling missing values | 02_titanic.py |
| Encoding text columns | 02_titanic.py, 03_house_prices.py |
| Handling outliers | 02_titanic.py, 03_house_prices.py |
| Normalising features | All examples |
| Creating new features | 03_house_prices.py |
| Training multiple models | 02_titanic.py, 03_house_prices.py |
| Comparing models | 02_titanic.py, 03_house_prices.py |
| Cross validation | 03_house_prices.py |
| Hyperparameter tuning | 03_house_prices.py |
| Saving models | 02_titanic.py, 03_house_prices.py |
| Drift detection | 03_house_prices.py |
