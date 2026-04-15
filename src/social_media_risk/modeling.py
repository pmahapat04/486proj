from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainResult:
    best_name: str
    best_pipeline: Pipeline
    test_metrics: dict[str, Any]
    y_test: np.ndarray
    y_pred: np.ndarray
    labels: list[str] | None


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = _split_columns(X)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Keep standardization for scale-sensitive models; trees ignore it.
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def _candidate_models(seed: int) -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
        ),
        "grad_boosting": GradientBoostingClassifier(random_state=seed),
    }


def _macro_f1_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    test_size: float,
    cv_folds: int,
) -> TrainResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    pre = build_preprocessor(X_train)
    models = _candidate_models(seed)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    best_name = None
    best_score = -1.0
    best_pipe = None

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        # sklearn's default scoring doesn't include macro F1 for multiclass by name
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    test_metrics = {
        "best_model": best_name,
        "cv_macro_f1_mean": best_score,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": _macro_f1_scorer(np.asarray(y_test), np.asarray(y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    labels = None
    try:
        labels = [str(x) for x in sorted(pd.unique(y))]
    except Exception:
        labels = None

    return TrainResult(
        best_name=best_name,
        best_pipeline=best_pipe,
        test_metrics=test_metrics,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        labels=labels,
    )

