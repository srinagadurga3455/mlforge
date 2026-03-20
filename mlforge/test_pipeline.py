import mlforge as mf

print("Step 1 — Generating test data...")
loader = mf.SyntheticLoader(task="regression", rows=500, features=8)
df     = loader.load()
print(f"  Data loaded: {df.shape}")

print("\nStep 2 — Validating data...")
mf.DataQualityCheck().check(df, target_col="target")

print("\nStep 3 — Cleaning data...")
df = mf.DataCleaner().clean(df)
df = mf.OutlierHandler(method="iqr", action="clip").fit_transform(df)

print("\nStep 4 — Normalising...")
norm = mf.Normalizer(method="minmax", exclude_cols=["target"])
df   = norm.fit_transform(df)

print("\nStep 5 — Training Random Forest...")
trainer = mf.Trainer(task="regression")
results = trainer.train(df, target_col="target", model_type="random_forest")
print(f"  Score: {results['score']}")

print("\nStep 6 — Training XGBoost...")
results2 = trainer.train(df, target_col="target", model_type="xgboost")
print(f"  Score: {results2['score']}")

print("\nStep 7 — Saving models...")
registry = mf.Registry(save_path="models/")
registry.save(results,  version="rf_v1")
registry.save(results2, version="xgb_v1")
registry.compare()

print("\nAll tests passed!")