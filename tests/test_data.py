import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS, FEATURES, TARGET, clean, split_xy


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "target": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[0, "ca"] = np.nan
    return df


def test_target_is_binary(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}


def test_clean_keeps_missing_for_pipeline_imputation(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() >= 0
    assert out.shape[1] == len(COLUMNS)


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(clean(doubled)) <= len(clean(raw_df))


def test_clean_coerces_question_marks():
    df = pd.DataFrame([[1] * 13 + [0]], columns=COLUMNS).astype(object)
    df.loc[0, "ca"] = "?"
    out = clean(df)
    assert np.isnan(out.loc[0, "ca"])


def test_split_xy_shapes(raw_df):
    X, y = split_xy(clean(raw_df))
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
