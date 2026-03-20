# Contributing to mlforge

## How to add a new data loader

1. Create a new file in `mlforge/data_sources/`
2. Inherit from `DataSource`
3. Implement `connect()`, `load()`, and `close()`
4. Register it in `mlforge/data_sources/__init__.py`
5. Add it to the master `mlforge/__init__.py`

```python
# mlforge/data_sources/my_loader.py

from .base import DataSource
import pandas as pd

class MyLoader(DataSource):
    def __init__(self, ...):
        super().__init__(name="my_loader")

    def connect(self):
        # open connection / validate config
        self.is_connected = True

    def load(self) -> pd.DataFrame:
        # load and return data
        return pd.DataFrame(...)

    def close(self):
        self.is_connected = False
```

## How to add a new model

1. Create a new file in `mlforge/models/`
2. Implement `fit()`, `predict()`, and optionally `predict_proba()`
3. Register it in `mlforge/models/__init__.py`
4. Add it to `MODEL_MAP` in `mlforge/model_training/trainer.py`
