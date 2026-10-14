import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import FEATURES
from src.features import add_age_bucket, build_preprocessor


def _frame(n=30, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "trestbps": rng.integers(100, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "thalach": rng.integers(90, 190, n).astype(float),
        "oldpeak": rng.random(n) * 4,
        "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n), "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    return df[FEATURES]


def test_preprocessor_handles_missing_and_scales():
    X = _frame()
    X.loc[0, "ca"] = np.nan
    X.loc[1, "chol"] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    assert abs(out[:, 0].mean()) < 1e-6  # age scaled


def test_age_bucket_feature_uses_expected_boundaries():
    X = pd.DataFrame({"age": [44, 45, 59, 60]})
    assert add_age_bucket(X)["age_bucket"].astype(str).tolist() == [
        "under_45", "45_to_59", "45_to_59", "60_plus"]


def test_preprocessor_ignores_unseen_category():
    prep = build_preprocessor().fit(_frame())
    new = _frame(5, seed=2)
    new["cp"] = 99
    assert prep.transform(new).shape[0] == 5


def test_pipeline_trains_and_predicts_probabilities():
    X = _frame(60)
    y = (X["age"] > 55).astype(int)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1)
