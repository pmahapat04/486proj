from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


_TARGET_GUESSES = {
    "addiction_level": [
        "addiction level",
        "addiction_level",
        "addictionlevel",
        "addiction",
        "social media addiction",
    ],
    "productivity_loss": [
        "productivity loss",
        "productivity_loss",
        "productivityloss",
        "productivity",
        "productivity score",
    ],
}


def _normalize_col(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch in {"_", " "})


def guess_target_column(df: pd.DataFrame, target: str) -> str:
    """
    Guess the target column from common variants.

    Raises:
        ValueError if no column can be confidently found.
    """
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in _TARGET_GUESSES.get(target, []):
        key = _normalize_col(cand)
        if key in normalized:
            return normalized[key]

    # Fallback: substring match
    for key, original in normalized.items():
        for cand in _TARGET_GUESSES.get(target, []):
            if _normalize_col(cand) in key or key in _normalize_col(cand):
                return original

    raise ValueError(
        f"Could not infer target column for {target!r}. "
        f"Available columns: {list(df.columns)}. "
        "Pass --target-col explicitly."
    )


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    target_col: str


def load_dataset(
    data_path: Path,
    *,
    target: str | None,
    target_col: str | None,
) -> Dataset:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Put the raw CSV under project/data/raw/ "
            "or pass the correct --data path."
        )

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Dataset at {data_path} is empty.")

    if target_col is None:
        if target is None:
            raise ValueError("Provide either --target or --target-col.")
        target_col = guess_target_column(df, target)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} not found. Available columns: {list(df.columns)}"
        )

    return Dataset(df=df, target_col=target_col)


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely missing."""
    keep = [c for c in df.columns if not df[c].isna().all()]
    return df[keep]


def coerce_object_numerics(df: pd.DataFrame, *, exclude: Iterable[str]) -> pd.DataFrame:
    """
    Best-effort conversion of object columns that look numeric.

    This is useful when CSVs store numeric values as strings (e.g. "3.5").
    """
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype != "object":
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            # Keep original dtype if conversion isn't possible.
            pass
    return df

