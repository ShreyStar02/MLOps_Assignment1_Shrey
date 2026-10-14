"""FastAPI serving layer: /predict, /health, /metrics with request logging."""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak",
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
META_PATH = MODEL_PATH.with_name("metadata.json")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])
ERRORS = Counter("api_prediction_errors_total", "Failed predictions")


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female")
    cp: int = Field(..., ge=1, le=4, description="chest pain type")
    trestbps: float = Field(..., ge=50, le=300, description="resting blood pressure")
    chol: float = Field(..., ge=50, le=700, description="serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250, description="max heart rate")
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float | None = Field(None, ge=0, le=3)
    thal: float | None = Field(None, description="3=normal, 6=fixed, 7=reversible")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class Prediction(BaseModel):
    risk_class: int
    risk_label: str
    confidence: float
    disease_probability: float


app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    return _model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        log.info("request method=%s path=%s status=%s duration_ms=%.1f",
                 request.method, path, status, elapsed * 1000)


@app.get("/health")
def health():
    try:
        get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.") from exc
    return {"status": "healthy", "service": "heart-disease-api"}


@app.get("/model-info")
def model_info():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"detail": "metadata unavailable"}


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures):
    try:
        model = get_model()
        row = pd.DataFrame([patient.model_dump()])[FEATURES]
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # noqa: BLE001
        ERRORS.inc()
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Unable to generate a prediction.") from exc
    pred = int(proba >= 0.5)
    label = "higher_risk" if pred else "lower_risk"
    PREDICTIONS.labels(label).inc()
    log.info("prediction label=%s probability=%.3f", label, proba)
    return Prediction(risk_class=pred, risk_label=label,
                      confidence=round(proba if pred else 1 - proba, 4),
                      disease_probability=round(proba, 4))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
