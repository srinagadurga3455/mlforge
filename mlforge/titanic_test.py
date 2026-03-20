"""
titanic_test.py
---------------
Full Titanic survival prediction pipeline using mlforge.

How to run:
    1. Download titanic.csv from:
       https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
    2. Place titanic.csv in the same folder as this file
    3. Open terminal in this folder and run:
       python titanic_test.py
"""

import mlforge as mf

print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION — mlforge")
print("=" * 55)

# ── Step 1 — Load data ────────────────────────────
print("\nStep 1 — Loading data...")
df = mf.FileLoader("titanic.csv").load()
print(f"  Loaded: {df.shape[0]} passengers, {df.shape[1]} columns")

# ── Step 2 — Drop useless columns ────────────────
print("\nStep 2 — Dropping useless columns...")
# Name, Ticket, PassengerId → too unique, model can't learn from them
# Cabin → 77% missing, not useful
df = df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"])
print(f"  Remaining columns: {df.columns.tolist()}")

# ── Step 3 — Validate data quality ───────────────
print("\nStep 3 — Checking data quality...")
mf.DataQualityCheck().check(df, target_col="Survived")
mf.MissingValueCheck(threshold=0.3).check(df)

# ── Step 4 — Fix data types ───────────────────────
print("\nStep 4 — Fixing data types...")
df = mf.TypeFixer().fix(df)

# ── Step 5 — Encode text columns ─────────────────
print("\nStep 5 — Encoding text columns...")
# Sex: male/female → 0/1
# Embarked: S/C/Q → 0/1/2
enc = mf.Encoder(method="label")
df  = enc.fit_transform(df, target_col="Survived")

# ── Step 6 — Fill missing values ─────────────────
print("\nStep 6 — Cleaning missing values...")
# Age: 20% missing → fill with median age
# Fare: 1% missing → fill with median fare
df = mf.DataCleaner(fill_numeric="median").clean(df)

# ── Step 7 — Handle outliers ──────────────────────
print("\nStep 7 — Handling outliers...")
df = mf.OutlierHandler(
    method       = "iqr",
    action       = "clip",          # cap extreme values, don't remove rows
    exclude_cols = ["Survived"]     # never touch the target column
).fit_transform(df)

# ── Step 8 — Normalise features ──────────────────
print("\nStep 8 — Normalising features...")
norm = mf.Normalizer(method="minmax", exclude_cols=["Survived"])
df   = norm.fit_transform(df)

print(f"\n  Final shape  : {df.shape}")
print(f"  Survival rate: {df['Survived'].mean()*100:.1f}%")

# ── Step 9 — Train and compare 5 models ──────────
print("\n" + "=" * 55)
print("  Training 5 models...")
print("=" * 55)

trainer = mf.Trainer(task="classification", test_size=0.2)
models  = [
    "random_forest",
    "decision_tree",
    "gradient_boosting",
    "logistic",
    "knn",
]
scores = {}

for model_name in models:
    print(f"\n  → {model_name}...")
    results          = trainer.train(df, target_col="Survived",
                                     model_type=model_name)
    scores[model_name] = results["score"]

# ── Step 10 — Compare results ─────────────────────
print("\n" + "=" * 55)
print("  MODEL COMPARISON")
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

# ── Step 11 — Save best model ─────────────────────
print("\n" + "=" * 55)
print("  Saving best model...")
print("=" * 55)

results  = trainer.train(df, target_col="Survived",
                          model_type=best_model)
registry = mf.Registry(save_path="models/")
name     = registry.save(results, version="titanic_v1")
registry.promote(name)

print(f"\n  Saved as   : models/{name}.pkl")
print(f"  Production : models/production_model.pkl")
print("\n✅ Done! Titanic pipeline complete.")
