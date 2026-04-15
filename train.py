#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from social_media_risk.runner import run_training


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train and evaluate models to predict social-media Addiction Level "
            "or Productivity Loss from structured features."
        )
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the input CSV dataset.",
    )
    p.add_argument(
        "--target",
        choices=["addiction_level", "productivity_loss"],
        default=None,
        help=(
            "Target to predict. Uses an automatic column guess unless "
            "--target-col is provided."
        ),
    )
    p.add_argument(
        "--target-col",
        default=None,
        help="Explicit target column name in the CSV (overrides --target).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction (default: 0.2).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write run artifacts (default: artifacts).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for model selection (default: 5).",
    )
    p.add_argument(
        "--binary-high-risk",
        action="store_true",
        help=(
            "Optional: collapse the label into a binary classification problem by "
            "mapping the highest risk category to 1 and all others to 0."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_training(
        data_path=args.data,
        target=args.target,
        target_col=args.target_col,
        test_size=args.test_size,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
        cv_folds=args.cv_folds,
        binary_high_risk=args.binary_high_risk,
    )


if __name__ == "__main__":
    main()

