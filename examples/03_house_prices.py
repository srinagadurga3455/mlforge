"""
03_house_prices.py
------------------
Predict house prices from features like size, bedrooms, location.

This is a REGRESSION problem:
  - Input  : house features (size, bedrooms, age, neighbourhood...)
  - Output : predicted price (a number)

What this example teaches:
  - The difference between regression and classification
  - How to create new features from existing ones (feature engineering)
  - How to use cross validation for more reliable scores
  - How to tune a model's hyperparameters automatically
  - How to detect data drift in production

Run it:
    python examples/03_house_prices.py
"""

import mlforge as mf
import pandas as pd
import numpy as np

print("=" * 55)
print("  HOUSE PRICE PREDICTION")
print("  Regression Example — mlforge")
print("=" * 55)

# ── Step 1: Generate house price data ────────────────────────────────────
# We generate synthetic data that mimics real house price datasets
# In a real project, you'd load your actual data here:
#   df = mf.FileLoader("house_prices.csv").load()

print("\nStep 1 — Creating house price dataset...")
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "sqft"         : np.random.normal(1800, 600, n).clip(400, 5000).astype(int),
    "bedrooms"     : np.random.choice([1,2,3,4,5], n, p=[0.05,0.20,0.40,0.25,0.10]),
    "bathrooms"    : np.random.choice([1,2,3,4],   n, p=[0.15,0.45,0.30,0.10]),
    "age_years"    : np.random.randint(0, 50, n),
    "garage"       : np.random.choice([0, 1], n, p=[0.3, 0.7]),
    "neighbourhood": np.random.choice(["downtown","suburb","rural"], n,
                                       p=[0.3, 0.5, 0.2]),
    "condition"    : np.random.choice(["poor","fair","good","excellent"], n,
                                       p=[0.1, 0.2, 0.5, 0.2]),
})

# Create a realistic price based on the features
price = (
    df["sqft"]      * 150
    + df["bedrooms"]  * 10000
    + df["bathrooms"] * 8000
    - df["age_years"] * 500
    + df["garage"]    * 15000
    + df["neighbourhood"].map({"downtown": 80000, "suburb": 20000, "rural": -10000})
    + df["condition"].map({"poor": -30000, "fair": -10000, "good": 10000, "excellent": 40000})
    + np.random.normal(0, 20000, n)   # add some noise
).clip(50000, 2000000)

df["price"] = price.astype(int)

print(f"  {len(df):,} houses")
print(f"  Price range: £{df['price'].min():,} — £{df['price'].max():,}")
print(f"  Average price: £{df['price'].mean():,.0f}")

# ── Step 2: Validate data quality ─────────────────────────────────────────
print("\nStep 2 — Checking data quality...")
mf.DataQualityCheck().check(df, target_col="price")

# ── Step 3: Feature engineering ───────────────────────────────────────────
# This is where human knowledge improves the model
# We create new columns that the model can learn from more easily

print("\nStep 3 — Creating new features...")
builder = mf.FeatureBuilder()

# Price per sqft is often more meaningful than raw sqft
# (not used as feature — just for understanding)

# Total rooms = bedrooms + bathrooms combined
df = builder.add_interaction(df, "bedrooms", "bathrooms",
                              name="total_rooms")

# Is the house large? (above average size)
df = builder.add_flag(df, "sqft", threshold=1800,
                       name="is_large_house")

# How new is the house? (0-10 years = new)
df = builder.add_flag(df, "age_years", threshold=10,
                       name="is_old_house")

print(f"  New features added: total_rooms, is_large_house, is_old_house")
print(f"  Total columns: {df.shape[1]}")

# ── Step 4: Encode text columns ───────────────────────────────────────────
# neighbourhood: downtown/suburb/rural → numbers
# condition: poor/fair/good/excellent → numbers

print("\nStep 4 — Encoding text columns...")
enc = mf.Encoder(method="label")
df  = enc.fit_transform(df, target_col="price")

# ── Step 5: Handle outliers ───────────────────────────────────────────────
# Some houses might have extreme prices
# We clip them instead of removing (keep all 1000 rows)

print("\nStep 5 — Handling outliers...")
df = mf.OutlierHandler(
    method       = "iqr",
    action       = "clip",
    exclude_cols = ["price"]
).fit_transform(df)

# ── Step 6: Normalise ─────────────────────────────────────────────────────
# sqft ranges 400–5000, age ranges 0–50
# Normalise so the model treats them equally

print("\nStep 6 — Normalising features...")
norm = mf.Normalizer(method="minmax", exclude_cols=["price"])
df   = norm.fit_transform(df)

# ── Step 7: Remove highly correlated features ─────────────────────────────
# If two columns are nearly identical, one is redundant
# FeatureSelector removes the weaker one automatically

print("\nStep 7 — Removing redundant features...")
selector = mf.FeatureSelector(method="correlation", threshold=0.95)
df       = selector.fit_transform(df, target_col="price")
print(f"  Features kept: {df.shape[1] - 1}")

# ── Step 8: Train models ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Training models...")
print("=" * 55)

trainer = mf.Trainer(task="regression", test_size=0.2)

models = ["random_forest", "gradient_boosting", "ridge", "xgboost"]
scores = {}

for model_name in models:
    print(f"\n  Training: {model_name}...")
    results          = trainer.train(df, target_col="price",
                                     model_type=model_name)
    scores[model_name] = results["score"]

# ── Step 9: Compare results ───────────────────────────────────────────────
# R² = how much of the price variation the model explains
#   1.0 = perfect prediction
#   0.8 = explains 80% of variation (very good)
#   0.5 = explains 50% of variation (mediocre)
#
# MAE = Mean Absolute Error = average £ off per prediction
#   MAE of 15000 means on average we're £15,000 off

print("\n" + "=" * 55)
print("  RESULTS — ranked by R²")
print("=" * 55)
print(f"{'Model':<22} {'R²':>8} {'MAE (£)':>12} {'RMSE (£)':>12}")
print("-" * 58)

for name, s in sorted(scores.items(),
                       key=lambda x: x[1]["r2"], reverse=True):
    print(f"{name:<22} {s['r2']:>8.4f} {s['mae']:>12,.0f} {s['rmse']:>12,.0f}")

best_model = max(scores.items(), key=lambda x: x[1]["r2"])[0]
best_score = scores[best_model]
print(f"\n  Best model : {best_model}")
print(f"  R²         : {best_score['r2']:.4f} "
      f"(explains {best_score['r2']*100:.1f}% of price variation)")
print(f"  MAE        : £{best_score['mae']:,.0f} average error per house")

# ── Step 10: Cross validation ─────────────────────────────────────────────
# A single train/test split can get lucky or unlucky
# Cross validation runs 5 different splits and averages the results
# This gives a much more reliable score

print("\n" + "=" * 55)
print("  Cross Validation (5 folds)...")
print("  More reliable than a single train/test split")
print("=" * 55)

results = trainer.train(df, target_col="price", model_type=best_model)
cv      = trainer.cross_validate(df, target_col="price", folds=5)
print(f"\n  Mean R²: {cv['mean']:.4f} ± {cv['std']:.4f}")
print(f"  All folds: {cv['all']}")
print(f"\n  ± {cv['std']:.4f} tells us how consistent the model is.")
print(f"  Lower std = more reliable predictions.")

# ── Step 11: Tune hyperparameters ─────────────────────────────────────────
# Hyperparameters are the model's settings (number of trees, depth, etc.)
# Instead of guessing, the Tuner tries many combinations automatically

print("\n" + "=" * 55)
print("  Hyperparameter Tuning...")
print("  Finding the best settings automatically")
print("=" * 55)

tuner   = mf.Tuner(model_type="random_forest", task="regression",
                    method="random", n_iter=10)
tuned   = tuner.tune(df, target_col="price")
print(f"\n  Best settings: {tuned['best_params']}")
print(f"  Best R²      : {tuned['best_score']:.4f}")

# ── Step 12: Save and monitor ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Saving model + setting up monitoring...")
print("=" * 55)

registry = mf.Registry(save_path="models/")
name     = registry.save(results, version="house_v1")
registry.promote(name)

# Set up drift detection
# This learns what "normal" house data looks like
# Later you can check if new listings look different
X_train = df.drop(columns=["price"])
detector = mf.DriftDetector(threshold=0.1)
detector.fit(X_train)
print(f"\n  Drift detector fitted.")
print(f"  Run detector.detect(new_data) to check for data drift.")

# Show feature importance
print("\n" + "=" * 55)
print("  Which features matter most?")
print("=" * 55)
v = mf.Validator(task="regression")
splits   = mf.DataSplitter(train_size=0.7, val_size=0.15,
                             test_size=0.15).split(df, target_col="price")
v.validate(results["model"],
           splits["X_train"], splits["X_test"],
           splits["y_train"], splits["y_test"])

print("\n✅ House prices example complete!")
print("\nKey takeaways:")
print("  1. Feature engineering (creating new columns) improves accuracy")
print("  2. Cross validation gives more reliable scores than one split")
print("  3. Hyperparameter tuning finds the best model settings")
print("  4. Drift detection catches when real-world data changes")
