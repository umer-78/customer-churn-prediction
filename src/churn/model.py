"""Train, compare and evaluate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .features import TARGET, build_preprocessor, prepare

MODELS = {
    "baseline (always stay)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "gradient boosting": GradientBoostingClassifier(random_state=42),
}


@dataclass
class Scores:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    confusion: list[list[int]] = field(default_factory=list)
    cv_roc_auc: float | None = None

    def row(self) -> str:
        cv = "     -" if self.cv_roc_auc is None else f"{self.cv_roc_auc:>6.3f}"
        return (f"{self.name:<22} {self.accuracy:>8.3f} {self.precision:>9.3f} {self.recall:>7.3f} "
                f"{self.f1:>6.3f} {self.roc_auc:>7.3f} {self.pr_auc:>7.3f} {self.brier:>6.3f} {cv}")


def evaluate(name: str, y_true, y_pred, y_prob) -> Scores:
    return Scores(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=average_precision_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def train_all(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, cv: int = 0):
    """Fit every candidate on the same split. Returns (scores, fitted pipelines, split)."""
    X = prepare(df)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    scores, fitted = [], {}
    for name, model in MODELS.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)
        s = evaluate(name, y_test, pipe.predict(X_test), prob)
        if cv:
            folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            s.cv_roc_auc = float(np.mean(cross_val_score(pipe, X, y, cv=folds, scoring="roc_auc")))
        scores.append(s)
        fitted[name] = pipe
    return scores, fitted, (X_train, X_test, y_train, y_test)


def top_features(pipeline: Pipeline, k: int = 12) -> pd.Series:
    """Feature importances (trees) or absolute coefficients (linear), by real name."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=names).sort_values(ascending=False).head(k)


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_frame(pipeline: Pipeline, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    prob = pipeline.predict_proba(prepare(df))[:, 1]
    band = pd.cut(prob, [-0.01, 0.33, 0.66, 1.01], labels=["low", "medium", "high"])
    out = pd.DataFrame({"churn_probability": prob.round(4), "risk": band.astype(str),
                        "predicted_churn": (prob >= threshold).astype(int)})
    if "customer_id" in df.columns:
        out.insert(0, "customer_id", df["customer_id"].to_numpy())
    return out
