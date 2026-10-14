"""Data loading and cleaning for the UCI Heart Disease dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def load_raw(path) -> pd.DataFrame:
    """Read the raw UCI file ('?' marks missing values)."""
    return pd.read_csv(Path(path), header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, binarise target (0 = healthy, 1 = disease), coerce dtypes.

    Missing values are intentionally kept (ca, thal); they are imputed inside the
    model pipeline so that imputation statistics are learnt on training folds only.
    """
    out = df.copy()
    for col in FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    out[TARGET] = (out[TARGET] > 0).astype(int)
    return out


def load_clean(path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def split_xy(df: pd.DataFrame):
    return df[FEATURES].replace({np.inf: np.nan}), df[TARGET]
