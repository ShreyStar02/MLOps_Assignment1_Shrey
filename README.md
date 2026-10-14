# Heart Disease Risk - End-to-End MLOps Project

Predicts heart disease risk from the UCI Heart Disease (Cleveland) dataset and serves it as a containerised, monitored FastAPI service.

Repository: `<PASTE GITHUB URL HERE>`  |  Full report: [docs/REPORT.md](docs/REPORT.md)

## Quick start (clean machine)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py     # downloads raw UCI data, writes data/processed/heart_clean.csv
python src/eda.py                   # figures -> reports/eda/
python src/train.py                 # compares LR, RF + GradientBoosting; tracks and exports the winner
pytest -v                           # unit tests
flake8 src api scripts tests        # lint
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

## Serve locally

```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
python scripts/sample_request.py http://localhost:8000 5
# or:
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Endpoints: `POST /predict`, `GET /health`, `GET /metrics` (Prometheus), `GET /model-info`, `GET /docs`.
The prediction response contains `risk_class`, `risk_label`, `confidence`, and `disease_probability`; the pipeline derives an age bucket from the supplied `age`.

## Monitoring (Prometheus + Grafana)

```bash
docker compose up --build
# API :8000   Prometheus :9090   Grafana :3000 (admin/admin) -> dashboard "Heart Disease API"
```

## Kubernetes (Docker Desktop / Minikube)

```bash
docker build -t heart-api:latest .
# Minikube only: minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
# Docker Desktop: http://localhost/health      Minikube: minikube service heart-api --url
kubectl apply -f k8s/ingress.yaml   # optional (needs ingress controller, host heart.local)
```

## Layout

| Path | Purpose |
|------|---------|
| `scripts/download_data.py` | Dataset acquisition + cleaning |
| `src/data.py`, `src/features.py` | Cleaning, imputation/scaling/encoding pipeline |
| `src/eda.py`, `src/train.py` | EDA figures; engineered features, three-model tuning, CV, MLflow tracking, model export |
| `api/app.py` | FastAPI service, logging, Prometheus metrics |
| `tests/` | Pytest unit tests (data, features, API) |
| `.github/workflows/ci.yml` | Lint -> test -> train -> docker build/smoke test -> artifacts |
| `k8s/`, `monitoring/`, `docker-compose.yml` | Deployment and monitoring config |
| `screenshots/` | Evidence for the report |
