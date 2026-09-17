"""Feature engineering and the preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"
NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges", "support_calls_6m",
    "late_payments_12m", "extra_services", "charges_per_month_of_tenure", "calls_per_year",
]
CATEGORICAL = ["contract", "internet_service", "payment_method", "tenure_bucket"]
BOOLEAN = ["paperless_billing", "senior_citizen", "has_dependents"]
FEATURE_COLUMNS = NUMERIC + CATEGORICAL + BOOLEAN


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Two ratios and a tenure bucket. Ratios beat raw totals for tree models."""
    out = df.copy()
    tenure = out["tenure_months"].clip(lower=1)
    out["charges_per_month_of_tenure"] = out["total_charges"].fillna(out["monthly_charges"] * tenure) / tenure
    out["calls_per_year"] = out["support_calls_6m"] * 2
    out["tenure_bucket"] = pd.cut(
        out["tenure_months"], bins=[-1, 6, 12, 24, 48, 1000],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
        ("bool", "passthrough", BOOLEAN),
    ])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw table in, model-ready feature frame out."""
    out = add_derived(df)
    for col in BOOLEAN:
        out[col] = out[col].astype(bool).astype(int)
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return out[FEATURE_COLUMNS]
