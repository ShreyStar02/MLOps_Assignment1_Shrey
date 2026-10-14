---
title: MLOps Assignment 1 Report
description: Rubric report for the heart disease classification MLOps assignment.
---

## MLOps Assignment 1 - Report

Student: `<NAME / BITS ID>`  |  Repo: [MLOps_Assignment1_Shrey](https://github.com/ShreyStar02/MLOps_Assignment1_Shrey)  |  Video: `<LINK>`

## 1. Problem and dataset

Binary classification of heart-disease presence using the UCI Heart Disease (Cleveland) data: 303 patients, 13 features + target. Original target (0-4) is binarised (0 = no disease, >0 = disease). Class balance: 164 healthy / 139 disease (about 54/46, no resampling needed). Missing values: `ca` (4) and `thal` (2). Data is fetched by `scripts/download_data.py`.

## 2. Setup

See [README.md](../README.md). All scripts run from a clean venv using `requirements.txt`.

## 3. EDA (reports/eda/)

Figures: class balance, histograms by target, correlation heatmap, categorical-vs-target counts, boxplots.

![class balance](../reports/eda/class_balance.png)
![histograms](../reports/eda/histograms.png)
![correlation](../reports/eda/correlation_heatmap.png)
![categorical features by target](../reports/eda/categorical_vs_target.png)
![numeric feature boxplots](../reports/eda/boxplots.png)

Observations (verify against your figures and edit): lower `thalach`, higher `oldpeak`, `exang=1`, `cp=4`, `ca>0` and `thal=7` associate with disease; `age`, `sex` also correlate. Features are weakly collinear, so no features were dropped.

## 4. Feature engineering and modelling

- Engineered `age_bucket` from age: under 45, 45-59, and 60+; the API accepts the original patient fields and the shared pipeline derives the bucket.
- Numeric (age, trestbps, chol, thalach, oldpeak): median imputation + StandardScaler.
- Categorical (sex, cp, fbs, restecg, exang, slope, ca, thal, age_bucket): most-frequent imputation + OneHotEncoder(handle_unknown=ignore).
- Imputers/scalers live inside the sklearn `Pipeline`, so they are fit on training folds only (no leakage) and are shipped with the model (identical preprocessing at serving time).
- Split: 80/20 stratified hold-out (seed 42). Tuning: GridSearchCV, 5-fold stratified CV, scoring ROC-AUC. LR grid: C in {0.01, 0.1, 1, 10}; RF grid: n_estimators {100, 300}, max_depth {3, 5, None}, min_samples_leaf {1, 3, 5}; GradientBoosting grid: n_estimators {100, 200}, learning_rate {0.03, 0.1}, max_depth {1, 2}.
- Metrics below are from the final `models/metadata.json` generated with scikit-learn 1.7.2. CV metrics are means across five folds; hold-out metrics use the untouched 20% split.

| Model | CV ROC-AUC | CV Accuracy | CV Precision | CV Recall | CV F1 | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|-----------:|------------:|-------------:|----------:|------:|-------------:|-------------:|---------------:|-----------:|--------:|
| Logistic Regression (C=0.1) | 0.901 | 0.835 | 0.857 | 0.774 | 0.810 | 0.965 | 0.885 | 0.839 | 0.929 | 0.881 |
| Random Forest | 0.901 | 0.822 | 0.824 | 0.783 | 0.800 | 0.951 | 0.869 | 0.813 | 0.929 | 0.867 |
| Gradient Boosting | 0.897 | 0.839 | 0.876 | 0.765 | 0.812 | 0.943 | 0.869 | 0.857 | 0.857 | 0.857 |

Selection: Logistic Regression was selected by the highest CV ROC-AUC (0.90146 vs 0.90127 for Random Forest), with best params `C=0.1`. The hold-out set contains 61 rows, so its estimates are noisy; this model is an educational demonstration, not a clinical tool.

![ROC curves](../reports/figures/roc_logistic_regression.png)
![Selected model confusion matrix](../reports/figures/cm_logistic_regression.png)
![Gradient Boosting feature importance](../reports/figures/importance_gradient_boosting.png)

## 5. Experiment tracking (MLflow)

Experiment `heart-disease-classification` (SQLite backend `mlflow.db`). Each candidate run logs best hyperparameters, seed, fold and split sizes, CV and test metrics, evaluation plots, and the fitted pipeline model. The clean-room run recreated the database because the previous ignored database referenced an inaccessible workstation path. No MLflow UI screenshots are currently present in `screenshots/`.

## 6. Packaging and reproducibility

`models/model.joblib` (full feature engineering, preprocessing + classifier pipeline), `models/metadata.json` (features, scikit-learn version, params, metrics, MLflow run id), `requirements.txt` and `requirements-serve.txt` (both pin scikit-learn 1.7.2), fixed random seeds.

## 7. CI/CD

GitHub Actions `.github/workflows/ci.yml`: checkout -> flake8 -> pytest with coverage -> download data -> train -> docker build -> container smoke test (`/health`, `/predict`) -> upload artifacts (train log, coverage, model, figures, mlflow.db). The workflow is configured but no CI run screenshots are present in `screenshots/`.

## 8. Containerisation

`Dockerfile` (python:3.11-slim, non-root `appuser`, HEALTHCHECK). Docker execution was not verified on this machine because the Docker CLI/daemon is unavailable; the image build, runtime, request logs, and non-root identity still need screenshot evidence.

## 9. Deployment

`k8s/deployment.yaml` (2 replicas, readiness/liveness probes, resource limits, LoadBalancer Service) and optional `k8s/ingress.yaml`. Kubernetes execution was not verified because `kubectl` is unavailable; pod readiness and LoadBalancer requests still need to be demonstrated on Docker Desktop.

## 10. Monitoring and logging

Every request is logged (method, path, status, latency); predictions logged with risk label and probability. Prometheus metrics: `api_requests_total`, `api_request_latency_seconds`, `api_predictions_total`, `api_prediction_errors_total`. Grafana dashboard is auto-provisioned. Docker Compose monitoring was not verified because Docker is unavailable; target-UP and dashboard screenshots remain outstanding.

## 11. Architecture

```mermaid
flowchart LR
  A[UCI dataset] --> B[download_data.py / clean]
  B --> C[EDA]
  B --> D[train.py: GridSearchCV + CV]
  D --> E[(MLflow tracking)]
  D --> F[models/model.joblib]
  F --> G[Docker image: FastAPI]
  H[GitHub Actions: lint, test, train, build] --> G
  G --> I[Kubernetes Deployment + LoadBalancer]
  I --> J[/predict]
  I --> K[/metrics] --> L[Prometheus] --> M[Grafana]
```

The diagram is included as Mermaid source; no architecture screenshot is currently available.

## 12. Limitations

Small dataset (303 rows, single hospital); no external validation; not a medical device. Possible future work: drift monitoring, model registry, automated retraining, canary deploys.
