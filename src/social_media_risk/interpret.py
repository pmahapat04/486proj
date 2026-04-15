from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TopFeature:
    name: str
    score: float


def _get_feature_names(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["pre"]
    # ColumnTransformer.get_feature_names_out is available in newer sklearn.
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def summarize_interpretability(pipe: Pipeline, *, top_k: int = 20) -> str:
    """
    Produce a readable feature-importance summary for the fitted pipeline.

    Supports:
      - LogisticRegression: coefficients
      - RandomForest / GradientBoosting: feature_importances_
    """
    model = pipe.named_steps["model"]
    names = _get_feature_names(pipe)

    lines: list[str] = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("")

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(getattr(model, "feature_importances_"))
        if names and len(names) == len(importances):
            pairs = list(zip(names, importances))
        else:
            pairs = [(f"feature_{i}", float(v)) for i, v in enumerate(importances)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"Top {min(top_k, len(pairs))} features by importance:")
        for n, v in pairs[:top_k]:
            lines.append(f"  {n}: {float(v):.6f}")
        return "\n".join(lines)

    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"))
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        if names and len(names) == coef.shape[1]:
            feat_names = names
        else:
            feat_names = [f"feature_{i}" for i in range(coef.shape[1])]

        if coef.shape[0] == 1:
            weights = coef[0]
            pos = sorted(zip(feat_names, weights), key=lambda x: x[1], reverse=True)[:top_k]
            neg = sorted(zip(feat_names, weights), key=lambda x: x[1])[:top_k]
            lines.append(f"Top {len(pos)} positive coefficients (push toward class 1):")
            for n, v in pos:
                lines.append(f"  {n}: {float(v):.6f}")
            lines.append("")
            lines.append(f"Top {len(neg)} negative coefficients (push toward class 0):")
            for n, v in neg:
                lines.append(f"  {n}: {float(v):.6f}")
            return "\n".join(lines)

        # Multiclass
        lines.append(f"Multiclass coefficients: {coef.shape[0]} classes")
        for k in range(coef.shape[0]):
            weights = coef[k]
            top = sorted(zip(feat_names, weights), key=lambda x: x[1], reverse=True)[:top_k]
            lines.append("")
            lines.append(f"Class index {k} top coefficients:")
            for n, v in top:
                lines.append(f"  {n}: {float(v):.6f}")
        return "\n".join(lines)

    return "\n".join(lines + ["No supported interpretability method found for this model."])

