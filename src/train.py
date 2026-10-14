"""Train, tune, cross-validate and track models with MLflow; export best pipeline."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import FEATURES, load_clean, split_xy  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

SEED = 42
DATA = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
plt.style.use("seaborn-v0_8-whitegrid")

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300],
         "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3, 5]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 200],
         "clf__learning_rate": [0.03, 0.1],
         "clf__max_depth": [1, 2]},
    ),
}


def holdout_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def save_plots(name, model, X_test, y_test, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="#007f86")
    ax.set_title(f"ROC - {name}")
    p = out_dir / f"roc_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="YlGnBu")
    ax.set_title(f"Confusion matrix - {name}")
    p = out_dir / f"cm_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = model.named_steps["prep"].get_feature_names_out()
        order = np.argsort(clf.feature_importances_)[::-1][:12]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([names[i] for i in order][::-1], clf.feature_importances_[order][::-1],
                color="#df6c4f")
        ax.set_title(f"Top feature importances - {name}")
        p = out_dir / f"importance_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    df = load_clean(DATA)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("heart-disease-classification")

    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Unbiased CV estimate of the tuned configuration on the training split
            cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            cv_metrics = {}
            for m in SCORING:
                cv_metrics[f"cv_{m}_mean"] = float(cvres[f"test_{m}"].mean())
                cv_metrics[f"cv_{m}_std"] = float(cvres[f"test_{m}"].std())
            test_metrics = holdout_metrics(best, X_test, y_test)

            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({"model": name, "seed": SEED, "cv_folds": 5,
                               "n_train": len(X_train), "n_test": len(X_test)})
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            for p in save_plots(name, best, X_test, y_test, REPORT_DIR / "figures"):
                mlflow.log_artifact(str(p), artifact_path="plots")
            mlflow.sklearn.log_model(best, artifact_path="model",
                                     serialization_format="cloudpickle")

            results[name] = {"model": best, "run_id": run.info.run_id,
                             "cv_roc_auc": cv_metrics["cv_roc_auc_mean"],
                             "metrics": {**cv_metrics, **test_metrics},
                             "params": {k: (v if isinstance(v, (int, float, str, type(None)))
                                            else str(v))
                                        for k, v in search.best_params_.items()}}
            print(f"{name}: CV ROC-AUC={cv_metrics['cv_roc_auc_mean']:.3f} "
                  f"test ROC-AUC={test_metrics['test_roc_auc']:.3f}")

    winner = max(results, key=lambda k: results[k]["cv_roc_auc"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(results[winner]["model"], MODEL_DIR / "model.joblib")
    meta = {"model": winner, "scikit_learn_version": sklearn.__version__,
            "features": FEATURES, "run_id": results[winner]["run_id"],
            "params": results[winner]["params"], "metrics": results[winner]["metrics"],
            "all_models": {k: v["metrics"] for k, v in results.items()}}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"Selected model: {winner} -> {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
