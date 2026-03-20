"""
02_titanic.py
-------------
Predict who survived the Titanic disaster.

This is a CLASSIFICATION problem:
  - Input  : passenger information (age, sex, class, fare...)
  - Output : survived (1) or did not survive (0)

What this example teaches:
  - How to handle real messy data
  - How to deal with missing values
  - How to encode text columns (male/female → 0/1)
  - How to compare multiple models
  - How to save the best model

Dataset:
  Download titanic.csv and place it in the same folder as this file.
  URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Run it:
    python examples/02_titanic.py
"""

import mlforge as mf

print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION")
print("  Classification Example — mlforge")
print("=" * 55)

# ── Step 1: Load the data ─────────────────────────────────────────────────
# The Titanic dataset has 891 passengers and 12 columns
# Each row = one passenger
# We want to predict the "Survived" column (0 or 1)

print("\nStep 1 — Loading data...")
df = mf.FileLoader("titanic.csv").load()
print(f"  {df.shape[0]} passengers, {df.shape[1]} columns")
print(f"  Survival rate: {df['Survived'].mean()*100:.1f}%")

# ── Step 2: Remove columns that won't help ────────────────────────────────
# Name  → every passenger has a unique name, model can't learn from it
# Ticket → same problem, too many unique values
# PassengerId → just a row number, meaningless
# Cabin → 77% missing, not enough data to be useful

print("\nStep 2 — Removing unhelpful columns...")
df = df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"])
print(f"  Kept: {df.columns.tolist()}")

# ── Step 3: Check data quality ────────────────────────────────────────────
# This catches problems BEFORE they break your model
# Things it checks:
#   - duplicate rows
#   - columns with all the same value (useless)
#   - skewed distributions
#   - class imbalance (one class much rarer than another)

print("\nStep 3 — Checking data quality...")
mf.DataQualityCheck().check(df, target_col="Survived")

# Also check for missing values
# Age has ~20% missing — that's ok, we'll fill it with the median age
mf.MissingValueCheck(threshold=0.3).check(df)

# ── Step 4: Fix data types ────────────────────────────────────────────────
# Sometimes numbers are stored as text ("30" instead of 30)
# TypeFixer detects and converts these automatically

print("\nStep 4 — Fixing data types...")
df = mf.TypeFixer().fix(df)

# ── Step 5: Encode text columns ───────────────────────────────────────────
# ML models only understand numbers. Text columns must be converted.
#
# Sex column:      male → 0,  female → 1
# Embarked column: S → 0,    C → 1,    Q → 2
#
# We use "label" encoding here (each category gets a number)
# For columns with many categories, use method="onehot" or method="target"

print("\nStep 5 — Encoding text columns (male/female → 0/1)...")
enc = mf.Encoder(method="label")
df  = enc.fit_transform(df, target_col="Survived")

# ── Step 6: Fill missing values ───────────────────────────────────────────
# Age: ~20% missing → fill with median age (middle value)
# Fare: 1% missing  → fill with median fare
# Using median instead of mean because it's more robust to outliers

print("\nStep 6 — Filling missing values...")
df = mf.DataCleaner(fill_numeric="median").clean(df)
# Also removes duplicate rows automatically

# ── Step 7: Handle outliers ───────────────────────────────────────────────
# Outliers = extreme values far from the rest of the data
# Example: most fares are £5–£50, but one passenger paid £512
# "clip" = cap the value at the boundary (don't remove the row)
# "iqr"  = uses the standard statistical method to define "extreme"

print("\nStep 7 — Handling outliers...")
df = mf.OutlierHandler(
    method       = "iqr",
    action       = "clip",
    exclude_cols = ["Survived"]   # never touch the target column
).fit_transform(df)

# ── Step 8: Normalise ─────────────────────────────────────────────────────
# Scale all numbers to 0–1 range
# Why? Age goes 0–80, Fare goes 5–500
# Without normalisation the model thinks Fare is more important
# just because the numbers are bigger — which is wrong

print("\nStep 8 — Normalising features to 0–1 range...")
norm = mf.Normalizer(method="minmax", exclude_cols=["Survived"])
df   = norm.fit_transform(df)

print(f"\n  Ready to train — {df.shape[0]} passengers, {df.shape[1]} columns")

# ── Step 9: Train and compare 5 different models ─────────────────────────
# We never assume one model is best — we try several and compare
# test_size=0.2 means 80% used for training, 20% held back for testing

print("\n" + "=" * 55)
print("  Training 5 models...")
print("=" * 55)

trainer = mf.Trainer(task="classification", test_size=0.2)

models = {
    "random_forest"    : "Many decision trees combined — usually very accurate",
    "decision_tree"    : "One tree — easy to explain, tends to overfit",
    "gradient_boosting": "Trees built one at a time, each fixing previous errors",
    "logistic"         : "Simple linear boundary — fast and interpretable",
    "knn"              : "Predicts based on the K most similar passengers",
}

scores = {}
for model_name, description in models.items():
    print(f"\n  Training: {model_name}")
    print(f"  ({description})")
    results          = trainer.train(df, target_col="Survived",
                                     model_type=model_name)
    scores[model_name] = results["score"]

# ── Step 10: Compare results ──────────────────────────────────────────────
# F1 score is the best metric for classification:
#   - Balances precision and recall
#   - Works well even when classes are slightly imbalanced
#   - 1.0 = perfect, 0.0 = useless

print("\n" + "=" * 55)
print("  RESULTS — ranked by F1 score")
print("=" * 55)
print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 62)

for name, s in sorted(scores.items(),
                       key=lambda x: x[1]["f1"], reverse=True):
    print(f"{name:<22} {s['accuracy']:>10.4f} {s['f1']:>8.4f} "
          f"{s['precision']:>10.4f} {s['recall']:>8.4f}")

best_model = max(scores.items(), key=lambda x: x[1]["f1"])[0]
print(f"\n  Best model : {best_model}")
print(f"  F1 score   : {scores[best_model]['f1']:.4f}")
print(f"  Accuracy   : {scores[best_model]['accuracy']:.4f}")
print(f"\n  Meaning: the model correctly predicts survival")
print(f"  {scores[best_model]['accuracy']*100:.1f}% of the time on data it has never seen.")

# ── Step 11: Save the best model ──────────────────────────────────────────
# Registry saves the model to disk so you can load it later
# without retraining — useful for deployment

print("\n" + "=" * 55)
print("  Saving best model...")
print("=" * 55)

results  = trainer.train(df, target_col="Survived", model_type=best_model)
registry = mf.Registry(save_path="models/")
name     = registry.save(results, version="titanic_v1")
registry.promote(name)

print(f"\n  Saved   : models/{name}.pkl")
print(f"  Production model: models/production_model.pkl")
print(f"\n  To load this model later:")
print(f"  model = registry.load('{name}')")
print(f"  predictions = model.predict(X_new)")

print("\n✅ Titanic example complete!")
