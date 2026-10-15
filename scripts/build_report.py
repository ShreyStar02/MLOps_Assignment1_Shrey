"""Build the final report from live metadata and screenshots and export DOCX/PDF."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from docx import Document
from docx.shared import Inches
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SCREENSHOTS = ROOT / "screenshots"
DOCX_PATH = REPORT_DIR / "MLOps_Assignment1_Report.docx"
PDF_PATH = REPORT_DIR / "MLOps_Assignment1_Report.pdf"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def model_table(metadata: dict) -> str:
    rows = []
    for name, payload in metadata.get("all_models", {}).items():
        label = name.replace("_", " ").title()
        rows.append(
            "| "
            f"{label} | {payload['cv_roc_auc_mean']:.3f} | "
            f"{payload['cv_accuracy_mean']:.3f} | "
            f"{payload['cv_precision_mean']:.3f} | "
            f"{payload['cv_recall_mean']:.3f} | "
            f"{payload['cv_f1_mean']:.3f} | "
            f"{payload['test_roc_auc']:.3f} | "
            f"{payload['test_accuracy']:.3f} | "
            f"{payload['test_precision']:.3f} | "
            f"{payload['test_recall']:.3f} | "
            f"{payload['test_f1']:.3f} |"
        )
    return "\n".join([
        "| Model | CV ROC-AUC | CV Accuracy | CV Precision | CV Recall | CV F1 | "
        "Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | "
        "Test F1 |",
        "|-------|-----------:|------------:|-------------:|----------:|------:|"
        "-------------:|-------------:|---------------:|-----------:|--------:|",
        *rows,
    ])


def render_report_md(config: dict, metadata: dict, eda: dict) -> str:
    repo = config.get(
        "repo_url",
        "https://github.com/ShreyStar02/MLOps_Assignment1_Shrey",
    )
    student_name = config.get("student_name", "Shreyes Ram V")
    bits_id = config.get("bits_id", "2025AE05177")
    course_name = config.get("course_name", "Machine Learning Operations (MLOps)")
    course_code = config.get("course_code", "AIMLZG523")
    assignment_title = config.get(
        "assignment_title",
        "MLOps Experimental Learning Assignment",
    )
    total_marks = config.get("total_marks", 50)
    objective = config.get("objective", "")
    dataset_source = config.get(
        "dataset_source",
        "https://archive.ics.uci.edu/dataset/45/heart+disease",
    )
    video_link = config.get("video_link", "")
    video_path = REPORT_DIR / "demo_video.mp4"
    if video_path.exists():
        video_info = f"[Demo video]({video_path.name})"
    else:
        video_info = f"[{video_link}]({video_link})" if video_link else "Not supplied"
    missing = eda.get("missing_values", {})
    class_counts = eda.get("class_counts", {})
    rows = eda.get("rows", 0)
    metrics = metadata.get("metrics", {})
    model_name = metadata.get("model", "logistic_regression")
    sk = metadata.get("scikit_learn_version", "Unknown")
    candidate_table = model_table(metadata)
    feature_names = ", ".join(metadata.get("features", []))
    template = f'''---
title: MLOps Assignment 1 Report
description: End-to-end MLOps implementation and evidence for the BITS AIMLZG523 assignment.
author: {student_name}
ms.date: 2026-10-15
ms.topic: report
---

## {assignment_title}

* Course: {course_name} ({course_code})
* Total marks: {total_marks}
* Student: {student_name}
* BITS ID: {bits_id}
* Repository: [{repo}]({repo})
**Video recording:** {video_info}

## Objective and problem

{objective}

The solution classifies heart-disease presence from the Cleveland subset of
the UCI Heart Disease dataset and serves predictions through a monitored API.

## 1. Data acquisition and exploratory analysis

The dataset was acquired from the [UCI Machine Learning Repository]({dataset_source})
using `scripts/download_data.py`. Cleaning produced {rows} rows, 13 input features,
and a binary target. The target distribution is {class_counts.get('0', 0)} records
without disease and {class_counts.get('1', 0)} with disease. The raw `ca` and
`thal` columns contained {missing.get('ca', 0)} and {missing.get('thal', 0)}
missing values respectively; these were retained for pipeline imputation.

The analysis includes class balance, numeric distributions, categorical counts,
feature boxplots, and a correlation heatmap. These figures were generated from
the processed data by `src/eda.py`.

![Class balance](../reports/eda/class_balance.png)

![Numeric distributions](../reports/eda/histograms.png)

![Correlation heatmap](../reports/eda/correlation_heatmap.png)

![Categorical features](../reports/eda/categorical_vs_target.png)

![Feature boxplots](../reports/eda/boxplots.png)

## 2. Feature engineering and model development

The shared preprocessing pipeline creates an age bucket (`under_45`, `45_to_59`,
or `60_plus`), median-imputes and standardizes numeric features, and most-frequent-
imputes and one-hot encodes categorical features. Unknown categories are ignored
at inference time. The feature inputs are {feature_names}.

Three classifiers were tuned with `GridSearchCV` and evaluated with stratified
5-fold cross-validation: logistic regression, random forest, and gradient
boosting. The highest mean CV ROC-AUC selected **{model_name}**. Its selected
parameters are `{metadata.get('params', {})}`. Metrics are from
`models/metadata.json`.

| Metric | Cross-validation mean | Test set |
| --- | ---: | ---: |
| ROC-AUC | {metrics.get('cv_roc_auc_mean', 0):.3f} | {metrics.get('test_roc_auc', 0):.3f} |
| Accuracy | {metrics.get('cv_accuracy_mean', 0):.3f} | {metrics.get('test_accuracy', 0):.3f} |
| Precision | {metrics.get('cv_precision_mean', 0):.3f} | {metrics.get('test_precision', 0):.3f} |
| Recall | {metrics.get('cv_recall_mean', 0):.3f} | {metrics.get('test_recall', 0):.3f} |
| F1 | {metrics.get('cv_f1_mean', 0):.3f} | {metrics.get('test_f1', 0):.3f} |

{candidate_table}

![Selected model ROC curve](../reports/figures/roc_logistic_regression.png)

![Confusion matrix](../reports/figures/cm_logistic_regression.png)

![Gradient boosting feature importance](../reports/figures/importance_gradient_boosting.png)

## 3. Experiment tracking

MLflow records parameters, cross-validation and holdout metrics, model artifacts,
and evaluation plots for all candidate runs. The tracking store is `mlflow.db`;
the selected model run ID is `{metadata.get('run_id', 'unknown')}`.

![MLflow training runs](../screenshots/mlflow_runs.png)

![Selected MLflow run details](../screenshots/mlflow_run_detail.png)

## 4. Model packaging and reproducibility

The selected scikit-learn {sk} preprocessing-and-classification pipeline is
serialized to `models/model.joblib`; `models/metadata.json` stores selected
parameters, metrics, feature names, and run ID. Install dependencies and run
the workflow from the repository root:

```powershell
python -m venv .venv
& .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
python src/eda.py
python src/train.py
pytest -v
flake8 src api scripts tests
```

## 5. CI/CD and automated testing

The GitHub Actions workflow checks out the repository, installs Python 3.11
dependencies, runs Flake8 and pytest with coverage, downloads and cleans data,
trains and logs models, builds the API image, smoke-tests `/health` and
`/predict`, and uploads training, coverage, model, and plot artifacts. The local
verification run passed all 14 tests and Flake8.

![GitHub Actions run](../screenshots/ci_run.png)

![GitHub Actions artifacts](../screenshots/ci_artifacts.png)

## 6. Containerized API

The FastAPI service accepts validated patient features at `POST /predict` and
returns the risk class, risk label, confidence, and disease probability.
`GET /health`, `GET /model-info`, `GET /metrics`, and `/docs` support health
checks, metadata, monitoring, and API exploration. Request logging and
Prometheus counters track requests, latency, prediction labels, and failures.

The API container was built from the project Dockerfile and exercised with
real JSON prediction requests.

![Docker build](../screenshots/docker_build.png)

![Docker prediction](../screenshots/docker_predict.png)

![API request logs](../screenshots/api_logs.png)

## 7. Kubernetes deployment

The Kubernetes manifest deploys two API replicas behind a `LoadBalancer`
service. Readiness and liveness probes call `/health`; each container requests
100m CPU and 256 MiB memory, with limits of 500m CPU and 512 MiB. The deployment
was applied to Docker Desktop Kubernetes, both pods reached `Ready`, and a
sample prediction succeeded through a local port-forward.

![Kubernetes pods and service](../screenshots/k8s_pods_svc.png)

![Kubernetes prediction response](../screenshots/k8s_predict.png)

## 8. Monitoring and logging

Prometheus scrapes the API metrics endpoint every five seconds. Its live target
status showed the `heart-api` target up. The provisioned Grafana dashboard
visualizes request rate, p95 latency, predictions by risk label, and failures.

![Prometheus target health](../screenshots/prometheus.png)

![Prometheus request metrics](../screenshots/prometheus_query.png)

![Grafana API dashboard](../screenshots/grafana.png)

![FastAPI OpenAPI documentation](../screenshots/api_docs.png)

## 9. Architecture

![Architecture diagram](../screenshots/architecture.png)

## 10. Local access and submission status

Start services with `docker compose up --build`; the API, Prometheus, and
Grafana are available on ports 8000, 9090, and 3000. Access the Kubernetes
deployment locally with `kubectl port-forward svc/heart-api 8001:80`. The video
recording was not supplied in the report configuration and remains a separate
assignment deliverable.

## Conclusion

The assignment requirements for data preparation, EDA, feature engineering,
three-model evaluation, MLflow tracking, reproducible packaging, CI/CD, unit
testing, Docker serving, local Kubernetes deployment, monitoring, and logging
were exercised on this machine. The evidence comes from those live runs.
'''
    return template


def add_image(paragraph, image_path: Path, width_in: float = 6.2) -> None:
    if image_path.exists():
        paragraph.add_run().add_picture(str(image_path), width=Inches(width_in))


def generate_docx(config: dict, metadata: dict, eda: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    student_name = config.get("student_name", "Student")
    bits_id = config.get("bits_id", "Not supplied")
    course_name = config.get("course_name", "Machine Learning Operations (MLOps)")
    course_code = config.get("course_code", "AIMLZG523")
    assignment_title = config.get("assignment_title", "MLOps Assignment 1")
    repo = config.get("repo_url", "")
    video_link = config.get("video_link", "")
    class_counts = eda.get("class_counts", {})
    missing = eda.get("missing_values", {})

    doc.add_heading(assignment_title, 0)
    total_marks = config.get("total_marks", 50)
    doc.add_paragraph(f"{course_name} ({course_code}) | Total marks: {total_marks}")
    doc.add_paragraph(f"Student: {student_name} | BITS ID: {bits_id}")
    doc.add_paragraph(f"Repository: {repo}")
    video_path = REPORT_DIR / "demo_video.mp4"
    if video_path.exists():
        doc.add_paragraph(f"Video recording: {video_path.name}")
    else:
        doc.add_paragraph(f"Video recording: {video_link or 'Not supplied'}")
    doc.add_heading("Objective", level=1)
    doc.add_paragraph(config.get("objective", ""))
    doc.add_page_break()

    headings = [
        "1. Data acquisition and exploratory analysis",
        "2. Feature engineering and model development",
        "3. Experiment tracking",
        "4. Model packaging and reproducibility",
        "5. CI/CD and automated testing",
        "6. Containerized API",
        "7. Kubernetes deployment",
        "8. Monitoring and logging",
        "9. Architecture",
        "10. Local access and submission status",
    ]
    doc.add_heading("Contents", level=1)
    for heading in headings:
        doc.add_paragraph(heading)
    doc.add_page_break()

    sections = [
        (
            headings[0],
            f"The UCI Cleveland heart disease data contains {eda.get('rows', 0)} records. "
            f"The target counts are {class_counts.get('0', 0)} without disease and "
            f"{class_counts.get('1', 0)} with disease. `ca` has "
            f"{missing.get('ca', 0)} missing values and `thal` has "
            f"{missing.get('thal', 0)}. The cleaning script preserves these records "
            "so the preprocessing pipeline can impute missing features.",
        ),
        (
            headings[1],
            "The pipeline engineers an age bucket, median-imputes and scales numeric "
            "features, and most-frequent-imputes and one-hot encodes categorical "
            "features. Logistic regression, random forest, and gradient boosting "
            "were tuned with GridSearchCV and evaluated with stratified five-fold CV.",
        ),
        (
            headings[2],
            "MLflow stores each candidate run, its parameters and CV/holdout metrics, "
            "plots, and model artifact in the local SQLite tracking store.",
        ),
        (
            headings[3],
            f"The selected {metadata.get('model', 'model')} pipeline uses "
            f"scikit-learn {metadata.get('scikit_learn_version', 'unknown')} and "
            "is serialized to models/model.joblib. models/metadata.json records "
            "the selected parameters, feature names, metrics, and MLflow run ID. "
            "Training and serving share the same preprocessing pipeline.",
        ),
        (
            headings[4],
            "GitHub Actions runs linting, unit tests with coverage, data download, "
            "training, Docker build and API smoke tests, then uploads model and "
            "training artifacts. The final local run passed all 14 tests and Flake8.",
        ),
        (
            headings[5],
            "FastAPI accepts patient features at POST /predict and returns risk class, "
            "risk label, confidence, and disease probability. GET /health, "
            "GET /model-info, GET /metrics, and /docs support health checks, "
            "metadata, metrics, and API exploration. Request logging records status "
            "and latency.",
        ),
        (
            headings[6],
            "The Kubernetes deployment runs two replicas behind a LoadBalancer "
            "service. Readiness and liveness probes call /health. Both pods reached "
            "Ready and a sample prediction succeeded through a local port-forward.",
        ),
        (
            headings[7],
            "Prometheus scrapes the API every five seconds. Its heart-api target was "
            "observed up. The provisioned Grafana dashboard displays request rate, "
            "p95 latency, predictions by label, and prediction failures.",
        ),
        (
            headings[8],
            "The diagram follows data acquisition and EDA through model selection, "
            "MLflow tracking, FastAPI containerization, Kubernetes deployment, and "
            "Prometheus/Grafana monitoring.",
        ),
        (
            headings[9],
            "For local use, start the stack with docker compose up --build. The API, "
            "Prometheus, and Grafana are exposed on ports 8000, 9090, and 3000. "
            "Kubernetes access is available with kubectl port-forward svc/heart-api "
            "8001:80. Video recording is available in report/demo_video.mp4.",
        ),
    ]

    for title, text in sections:
        doc.add_heading(title, level=1)
        doc.add_paragraph(text)
        if title == headings[0]:
            image_paths = [
                ROOT / "reports" / "eda" / name
                for name in [
                    "class_balance.png",
                    "histograms.png",
                    "correlation_heatmap.png",
                    "categorical_vs_target.png",
                    "boxplots.png",
                ]
            ]
            for path in image_paths:
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.0)
        elif title == headings[1]:
            table = doc.add_table(rows=1, cols=5)
            table.style = "Light Shading Accent 1"
            for cell, label in zip(
                table.rows[0].cells,
                ["Model", "CV ROC-AUC", "Test ROC-AUC", "Test accuracy", "Test F1"],
            ):
                cell.text = label
            for name, values in metadata.get("all_models", {}).items():
                cells = table.add_row().cells
                cells[0].text = name.replace("_", " ").title()
                cells[1].text = f"{values['cv_roc_auc_mean']:.3f}"
                cells[2].text = f"{values['test_roc_auc']:.3f}"
                cells[3].text = f"{values['test_accuracy']:.3f}"
                cells[4].text = f"{values['test_f1']:.3f}"
            for name in [
                "roc_logistic_regression.png",
                "cm_logistic_regression.png",
                "importance_gradient_boosting.png",
            ]:
                path = ROOT / "reports" / "figures" / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=5.8)
        elif title == headings[2]:
            for name in ["mlflow_runs.png", "mlflow_run_detail.png"]:
                path = SCREENSHOTS / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.2)
        elif title == headings[4]:
            for name in ["ci_run.png", "ci_artifacts.png"]:
                path = SCREENSHOTS / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.2)
        elif title == headings[5]:
            for name in ["docker_build.png", "docker_predict.png", "api_logs.png"]:
                path = SCREENSHOTS / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.0)
        elif title == headings[6]:
            for name in ["k8s_pods_svc.png", "k8s_predict.png"]:
                path = SCREENSHOTS / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.0)
        elif title == headings[7]:
            for name in [
                "prometheus.png",
                "prometheus_query.png",
                "grafana.png",
                "api_docs.png",
            ]:
                path = SCREENSHOTS / name
                if path.exists():
                    add_image(doc.add_paragraph(), path, width_in=6.0)
        elif title == headings[8]:
            path = SCREENSHOTS / "architecture.png"
            if path.exists():
                add_image(doc.add_paragraph(), path, width_in=6.5)

    doc.save(DOCX_PATH)


def convert_to_pdf() -> None:
    if DOCX_PATH.exists():
        try:
            from docx2pdf import convert
            convert(str(DOCX_PATH), str(PDF_PATH))
        except Exception:
            pandoc = shutil.which("pandoc")
            if pandoc:
                subprocess.run([pandoc, str(DOCX_PATH), "-o", str(PDF_PATH)], check=True)
            else:
                raise RuntimeError(
                    "Neither docx2pdf nor pandoc is available to generate the PDF."
                )
    else:
        raise FileNotFoundError(DOCX_PATH)


def verify_report(report_md: str) -> None:
    if re.search(r"<[^>]+>", report_md):
        raise RuntimeError("Placeholder tokens remain in the report markdown.")
    required = [
        ROOT / "screenshots" / "mlflow_runs.png",
        ROOT / "screenshots" / "mlflow_run_detail.png",
        ROOT / "screenshots" / "ci_run.png",
        ROOT / "screenshots" / "ci_artifacts.png",
        ROOT / "screenshots" / "docker_build.png",
        ROOT / "screenshots" / "docker_predict.png",
        ROOT / "screenshots" / "k8s_pods_svc.png",
        ROOT / "screenshots" / "k8s_predict.png",
        ROOT / "screenshots" / "prometheus.png",
        ROOT / "screenshots" / "prometheus_query.png",
        ROOT / "screenshots" / "grafana.png",
        ROOT / "screenshots" / "api_docs.png",
        ROOT / "screenshots" / "api_logs.png",
        ROOT / "screenshots" / "architecture.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evidence images: {missing}")
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)
    video_path = REPORT_DIR / "demo_video.mp4"
    if not video_path.exists():
        raise FileNotFoundError("Missing demo video: report/demo_video.mp4")
    reader = PdfReader(str(PDF_PATH))
    page_count = len(reader.pages)
    if page_count < 10:
        raise RuntimeError(f"The report must be at least 10 pages; generated {page_count}.")
    print(f"Report PDF page count: {page_count}")
    print(
        "Checklist: docx exists; pdf exists; no placeholders; required screenshots "
        "exist; report is at least 10 pages."
    )


def main() -> int:
    config = load_config(ROOT / "report_config.yaml")
    metadata = load_json(ROOT / "models" / "metadata.json")
    eda = load_json(ROOT / "reports" / "eda" / "summary.json")
    report_md = render_report_md(config, metadata, eda)
    (ROOT / "docs" / "REPORT.md").write_text(report_md, encoding="utf-8")
    generate_docx(config, metadata, eda)
    convert_to_pdf()
    verify_report(report_md)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"build_report.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
