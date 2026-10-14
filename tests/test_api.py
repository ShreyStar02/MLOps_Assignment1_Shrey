import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib

from src.data import FEATURES
from src.features import build_preprocessor

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
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
    })[FEATURES]
    y = (X["age"] > 55).astype(int)
    model = Pipeline([("prep", build_preprocessor()),
                      ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump(model, path)

    import api.app as app_module
    app_module.MODEL_PATH = path
    app_module.META_PATH = path.with_name("metadata.json")
    app_module._model = None
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/health").json() == {
        "status": "healthy", "service": "heart-disease-api"}


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_class"] in (0, 1)
    assert body["risk_label"] in ("higher_risk", "lower_risk")
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0 <= body["disease_probability"] <= 1


def test_predict_accepts_missing_optional_fields(client):
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_invalid_input(client):
    bad = {**SAMPLE, "age": -5}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=missing).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
