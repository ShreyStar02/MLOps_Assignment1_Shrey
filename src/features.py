"""Preprocessing pipeline shared by training and serving."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import pandas as pd

from src.data import CATEGORICAL, NUMERIC


def add_age_bucket(X: pd.DataFrame) -> pd.DataFrame:
    transformed = X.copy()
    transformed["age_bucket"] = pd.cut(
        transformed["age"],
        bins=[0, 45, 60, float("inf")],
        labels=["under_45", "45_to_59", "60_plus"],
        right=False,
    )
    return transformed


def _age_bucket_feature_names_out(transformer, input_features):
    return list(input_features) + ["age_bucket"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL + ["age_bucket"]),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(
            add_age_bucket, validate=False,
            feature_names_out=_age_bucket_feature_names_out)),
        ("columns", columns),
    ])
