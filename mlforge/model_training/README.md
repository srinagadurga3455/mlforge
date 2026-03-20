# model_training — Train, tune, and save models

## Why does this folder exist?

Having a model class is not enough. You need to:
1. Split your data correctly (train/test)
2. Fit the model
3. Score it
4. Find the best settings (hyperparameters)
5. Save the model so you can use it later
6. Version it so you can compare v1 vs v2

This folder handles all of that.

## Files

| File | What it does |
|---|---|
| `trainer.py` | Trains any model — handles split, fit, score in one call |
| `tuner.py` | Automatically finds the best hyperparameters |
| `registry.py` | Saves, loads, versions, and compares trained models |

## The ML concept: train/test split

You NEVER evaluate a model on data it was trained on.
Why? Because the model already saw that data — it would get an
unfairly high score. Like a student being tested on questions
they already know the answers to.

The standard split is:
- **80% training data** → model learns from this
- **20% test data** → model is evaluated on this (never seen during training)

```python
trainer = Trainer(task="regression", test_size=0.2)
results = trainer.train(df, target_col="price")
# Automatically splits 80/20, trains, and returns score
```

## The ML concept: hyperparameters

A model's hyperparameters are its settings — things you choose
before training, like:
- How many trees in a Random Forest? (n_estimators)
- How deep can each tree grow? (max_depth)
- How fast should it learn? (learning_rate)

The wrong settings can make a great model perform poorly.
The Tuner tries many combinations and finds the best ones automatically.

```python
tuner   = Tuner(model_type="random_forest", task="regression")
results = tuner.tune(df, target_col="price")
print(results["best_params"])   # {"n_estimators": 200, "max_depth": 10}
```

## The ML concept: model versioning

In production, you train new versions of your model over time.
The Registry lets you save each version, compare them, and
promote the best one to production.

```python
registry = Registry(save_path="models/")
registry.save(results, version="v1")
registry.save(results_v2, version="v2")
registry.compare()                      # see which version scored better
registry.promote("random_forest_v2")    # use v2 in production
```
