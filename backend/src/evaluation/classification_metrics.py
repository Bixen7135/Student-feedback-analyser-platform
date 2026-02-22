"""Classification metrics: macro F1, per-class F1, confusion matrices, stratification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: np.ndarray
    support: dict[str, int]
    classes: list[str]
    n_samples: int


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> ClassificationMetrics:
    """Compute full classification metrics for a single task."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    present_classes = [c for c in classes if c in y_true or c in y_pred]
    if not present_classes:
        present_classes = classes

    macro_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    per_class_f1 = {
        c: float(f1_score(y_true, y_pred, labels=[c], average="micro", zero_division=0))
        for c in present_classes
    }
    per_class_precision = {
        c: float(precision_score(y_true, y_pred, labels=[c], average="micro", zero_division=0))
        for c in present_classes
    }
    per_class_recall = {
        c: float(recall_score(y_true, y_pred, labels=[c], average="micro", zero_division=0))
        for c in present_classes
    }
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    support = {c: int((y_true == c).sum()) for c in present_classes}

    return ClassificationMetrics(
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_matrix=cm,
        support=support,
        classes=present_classes,
        n_samples=len(y_true),
    )


def stratify_metrics_by(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    stratify_col: str,
    classes: list[str],
) -> dict[str, ClassificationMetrics]:
    """Compute classification metrics stratified by a column (e.g., language)."""
    results: dict[str, ClassificationMetrics] = {}
    for stratum in df[stratify_col].dropna().unique():
        mask = df[stratify_col] == stratum
        subset = df[mask]
        if len(subset) < 2:
            continue
        metrics = compute_classification_metrics(
            subset[y_true_col].values,
            subset[y_pred_col].values,
            classes,
        )
        results[str(stratum)] = metrics
    return results


def metrics_to_dict(m: ClassificationMetrics) -> dict[str, Any]:
    """Serialize ClassificationMetrics to a JSON-safe dict."""
    return {
        "macro_f1": m.macro_f1,
        "accuracy": m.accuracy,
        "per_class_f1": m.per_class_f1,
        "per_class_precision": m.per_class_precision,
        "per_class_recall": m.per_class_recall,
        "confusion_matrix": m.confusion_matrix.tolist(),
        "support": m.support,
        "classes": m.classes,
        "n_samples": m.n_samples,
    }
