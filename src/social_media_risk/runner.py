from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from social_media_risk.data import (
    coerce_object_numerics,
    drop_all_null_columns,
    load_dataset,
)
from social_media_risk.interpret import summarize_interpretability
from social_media_risk.modeling import train_and_select


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _maybe_make_binary_high_risk(y: pd.Series) -> pd.Series:
    """
    Map the highest-risk category to 1 and everything else to 0.

    Works for:
      - numeric ordinal labels (max value is "highest risk")
      - string labels (lexicographic max after normalization, as a fallback)
    """
    if y.nunique() <= 2:
        return y

    if pd.api.types.is_numeric_dtype(y):
        high = y.max()
        return (y == high).astype(int)

    # Try to interpret common text levels.
    normalized = y.astype(str).str.strip().str.lower()
    for token in ["high", "severe", "very high", "addicted"]:
        if token in set(normalized):
            return (normalized == token).astype(int)

    # Fallback: pick the most "max" label to serve as high-risk.
    high = sorted(set(normalized))[-1]
    return (normalized == high).astype(int)


def run_training(
    *,
    data_path: Path,
    target: str | None,
    target_col: str | None,
    test_size: float,
    seed: int,
    artifacts_dir: Path,
    cv_folds: int,
    binary_high_risk: bool,
) -> None:
    ds = load_dataset(data_path, target=target, target_col=target_col)
    df = drop_all_null_columns(ds.df)
    df = coerce_object_numerics(df, exclude=[ds.target_col])

    y = df[ds.target_col]
    X = df.drop(columns=[ds.target_col])

    # Drop an obvious index column if present.
    for cand in ["Unnamed: 0", "index", "Index", "id", "ID"]:
        if cand in X.columns and X[cand].nunique() == len(X):
            X = X.drop(columns=[cand])
            break

    if binary_high_risk:
        y = _maybe_make_binary_high_risk(y)

    result = train_and_select(
        X,
        y,
        seed=seed,
        test_size=test_size,
        cv_folds=cv_folds,
    )

    run_dir = artifacts_dir / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(result.best_pipeline, run_dir / "best_model.joblib")

    # Save metrics
    (run_dir / "metrics.json").write_text(json.dumps(result.test_metrics, indent=2))

    # Confusion matrix plot
    cm = np.asarray(result.test_metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Interpretability summary
    summary = summarize_interpretability(result.best_pipeline, top_k=25)
    (run_dir / "top_features.txt").write_text(summary)

    print(f"Wrote artifacts to: {run_dir}")
    print(json.dumps({k: result.test_metrics[k] for k in ['best_model','cv_macro_f1_mean','test_accuracy','test_macro_f1']}, indent=2))

