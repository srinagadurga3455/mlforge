# evaluation — Measure how good your model actually is

## Why does this folder exist?

Training a model is not the end. You need to answer:
- How accurate is it?
- Where does it fail?
- Is it overfitting?
- Which features matter most?

This folder gives you all those answers.

## Files

| File | What it does |
|---|---|
| `metrics.py` | Calculate accuracy, R², F1, ROC-AUC, confusion matrix, and more |
| `validator.py` | Check for overfitting, show feature importance, prediction stats |

## Regression metrics explained

| Metric | What it means | Good value |
|---|---|---|
| R² | How much variance the model explains (1.0 = perfect) | > 0.8 is good |
| MAE | Average error in original units (e.g. £1,500 off) | Lower = better |
| RMSE | Like MAE but punishes big errors more | Lower = better |
| MAPE | Average % error | < 10% is good |

## Classification metrics explained

| Metric | What it means | When it matters |
|---|---|---|
| Accuracy | % of correct predictions | Balanced classes |
| Precision | Of all positive predictions, how many were right | When false positives are costly (spam filter) |
| Recall | Of all actual positives, how many did we catch | When false negatives are costly (cancer detection) |
| F1 | Balance between precision and recall | Imbalanced classes |
| ROC-AUC | How well model separates classes | Overall ranking ability |

## Example

```python
from mlforge.evaluation import Metrics, Validator

# Calculate all metrics
calc = Metrics(task="classification")
calc.calculate(y_test, y_pred, y_proba=trainer.predict_proba(X_test))

# Check model health
Validator(task="regression").validate(
    model, X_train, X_test, y_train, y_test
)
# Prints: HEALTHY or OVERFITTING, top features, prediction stats
```

## Learning ML concept: the confusion matrix

For classification, the confusion matrix shows:
```
                Predicted: No    Predicted: Yes
Actual: No         950               50         ← False Positives (FP)
Actual: Yes         20              980         ← True Positives (TP)
```
- High diagonal numbers = good model
- High off-diagonal numbers = model is confused
