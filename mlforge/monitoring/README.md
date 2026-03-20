# monitoring — Watch your model after it's deployed

## Why does this folder exist?

Most ML tutorials end at "evaluate the model". But in the real world,
you deploy the model and it runs for months or years.

The world changes. Your data changes. Your model gets worse over time
without you realising it.

This folder helps you catch that.

## The ML concept: model drift

Imagine you train a house price model in January using data from 2023.
By June, interest rates have changed, a new neighbourhood opened up,
and the economy shifted.

The patterns your model learned from 2023 data no longer apply.
Predictions become less and less accurate. This is called **drift**.

There are two types:
- **Data drift** — the input features start looking different
- **Concept drift** — the relationship between features and target changes

## Files

| File | What it does |
|---|---|
| `drift_detector.py` | Alerts when production data looks different from training data |
| `prediction_logger.py` | Logs every prediction so you can audit and analyse them |

## Example

```python
from mlforge.monitoring import DriftDetector, PredictionLogger

# Learn what normal data looks like during training
detector = DriftDetector(threshold=0.1)
detector.fit(X_train)

# Every week, check if new data looks different
result = detector.detect(X_new)
if result["drift_detected"]:
    print("Data has drifted! Retrain your model.")
    print("Changed columns:", result["drifted_columns"])

# Log every prediction in production
pl = PredictionLogger(log_path="logs/")
pl.log(input_data={"age": 35, "income": 50000}, prediction=180000)
pl.summary()   # daily statistics
```

## When should you retrain?

- Performance drops below a threshold (e.g. accuracy < 80%)
- Drift detected in key features
- New labelled data is available
- Significant time has passed (monthly retraining is common)

## Learning ML concept: why logging predictions matters

If your model makes a bad prediction and someone complains,
you need to be able to go back and see: what data did the model
receive? What did it predict? When?

Without prediction logging you have no audit trail.
With prediction logging you can investigate any prediction ever made.
