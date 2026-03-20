"""
01_quickstart.py
----------------
The simplest possible mlforge pipeline.

What this example teaches:
  - How to load data
  - How to clean it
  - How to train a model
  - How to evaluate it
  - The whole thing in under 30 lines

Run it:
    python examples/01_quickstart.py
"""

import mlforge as mf

# ── Step 1: Generate some data to work with ───────────────────────────────
# SyntheticLoader creates fake data instantly — perfect for testing
# task="regression" means we want to predict a number
loader = mf.SyntheticLoader(task="regression", rows=500, features=6)
df     = loader.load()
# df now has 6 feature columns and 1 target column called "target"

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ── Step 2: Clean the data ────────────────────────────────────────────────
# Fill any missing values and remove duplicates
df = mf.DataCleaner().clean(df)

# Scale all numbers to the same range (0 to 1)
# This prevents columns with big numbers from dominating the model
norm = mf.Normalizer(method="minmax", exclude_cols=["target"])
df   = norm.fit_transform(df)

# ── Step 3: Train a model ─────────────────────────────────────────────────
# Trainer handles the train/test split automatically (80% train, 20% test)
trainer = mf.Trainer(task="regression")
results = trainer.train(df, target_col="target", model_type="random_forest")

# ── Step 4: See the results ───────────────────────────────────────────────
print("\nModel results:")
print(f"  R²   = {results['score']['r2']}   (1.0 = perfect, 0 = useless)")
print(f"  MAE  = {results['score']['mae']}  (average prediction error)")
print(f"  RMSE = {results['score']['rmse']} (penalises big errors more)")

# ── Step 5: Make a prediction on new data ────────────────────────────────
import pandas as pd
new_data   = df.drop(columns=["target"]).head(3)
prediction = trainer.predict(new_data)
print(f"\nPredictions on 3 new rows: {prediction.round(2)}")

print("\nDone! That's all it takes to train a model with mlforge.")
