"""Robustness evaluation: performance on short/empty text and sensitivity analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.text_tasks.base import TextClassifier
from src.evaluation.classification_metrics import compute_classification_metrics, ClassificationMetrics


def evaluate_on_short_text(
    df: pd.DataFrame,
    clf: TextClassifier,
    text_col: str,
    label_col: str,
    char_threshold: int = 20,
) -> ClassificationMetrics | None:
    """Evaluate classifier performance on responses with very short text."""
    mask = df[text_col].fillna("").str.len() <= char_threshold
    subset = df[mask]
    if len(subset) < 2:
        return None
    preds = clf.predict(subset[text_col].fillna("").tolist())
    return compute_classification_metrics(
        subset[label_col].values, preds, clf.classes_
    )


def evaluate_on_empty_text(
    df: pd.DataFrame,
    clf: TextClassifier,
    text_col: str,
    label_col: str,
) -> ClassificationMetrics | None:
    """Evaluate classifier performance on empty text responses."""
    mask = df[text_col].fillna("").str.strip() == ""
    subset = df[mask]
    if len(subset) < 1:
        return None
    preds = clf.predict([""] * len(subset))
    if len(subset) < 2:
        # Cannot compute metrics with single sample — return prediction summary
        return None
    return compute_classification_metrics(
        subset[label_col].values, preds, clf.classes_
    )


def sensitivity_analysis(
    df: pd.DataFrame,
    clf: TextClassifier,
    text_col: str,
    label_col: str,
    group_col: str,
) -> dict[str, Any]:
    """
    Run classifier on each stratum of group_col and collect metrics.
    Used for sensitivity analysis by language and detail level.
    """
    results: dict[str, Any] = {}
    for group_val in df[group_col].dropna().unique():
        subset = df[df[group_col] == group_val]
        if len(subset) < 2:
            continue
        preds = clf.predict(subset[text_col].fillna("").tolist())
        metrics = compute_classification_metrics(
            subset[label_col].values, preds, clf.classes_
        )
        results[str(group_val)] = {
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "n_samples": metrics.n_samples,
        }
    return results
