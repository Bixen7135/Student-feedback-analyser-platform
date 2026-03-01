"""Performance diagnostics for classification and regression outputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classification_diagnostics(
    y_true: pd.Series | list[Any] | np.ndarray,
    y_pred: pd.Series | list[Any] | np.ndarray,
    confidences: pd.Series | list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return confusion matrix, per-class metrics, and confidence calibration."""
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).fillna("").astype(str),
            "y_pred": pd.Series(y_pred).fillna("").astype(str),
        }
    )
    if confidences is not None:
        frame["confidence"] = pd.to_numeric(pd.Series(confidences), errors="coerce")
    frame = frame[
        (frame["y_true"].str.strip() != "") & (frame["y_pred"].str.strip() != "")
    ].copy()

    if frame.empty:
        return {
            "n_rows": 0,
            "labels": labels or [],
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [],
            "per_class": {},
            "calibration": [],
            "ece": 0.0,
        }

    all_labels = labels or sorted(
        set(frame["y_true"].tolist()) | set(frame["y_pred"].tolist())
    )
    confusion = pd.crosstab(
        pd.Categorical(frame["y_true"], categories=all_labels),
        pd.Categorical(frame["y_pred"], categories=all_labels),
        dropna=False,
    ).reindex(index=all_labels, columns=all_labels, fill_value=0)

    total = int(confusion.to_numpy().sum())
    correct = int(np.trace(confusion.to_numpy()))
    accuracy = round(correct / total, 4) if total else 0.0

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in all_labels:
        tp = int(confusion.loc[label, label])
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        support = int(confusion.loc[label].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    calibration = _confidence_calibration(frame, n_bins=n_bins)
    return {
        "n_rows": int(len(frame)),
        "labels": list(all_labels),
        "accuracy": accuracy,
        "macro_f1": round(float(np.mean(f1_values)) if f1_values else 0.0, 4),
        "confusion_matrix": confusion.values.tolist(),
        "per_class": per_class,
        "calibration": calibration["bins"],
        "ece": calibration["ece"],
    }


def regression_diagnostics(
    y_true: pd.Series | list[float] | np.ndarray,
    y_pred: pd.Series | list[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return residual summaries and binned fitted-vs-residual patterns."""
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(pd.Series(y_true), errors="coerce"),
            "y_pred": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return {
            "n_rows": 0,
            "residual_summary": {},
            "residual_vs_fitted": [],
            "abs_residual_fitted_correlation": 0.0,
        }

    frame["residual"] = frame["y_true"] - frame["y_pred"]
    residuals = frame["residual"]
    fitted = frame["y_pred"]

    residual_summary = {
        "mean": round(float(residuals.mean()), 4),
        "std": round(float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0, 4),
        "mae": round(float(residuals.abs().mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
        "min": round(float(residuals.min()), 4),
        "median": round(float(residuals.median()), 4),
        "max": round(float(residuals.max()), 4),
    }

    bins = _residual_bins(frame, n_bins=n_bins)
    abs_corr = fitted.corr(residuals.abs(), method="pearson")
    if pd.isna(abs_corr):
        abs_corr = 0.0

    return {
        "n_rows": int(len(frame)),
        "residual_summary": residual_summary,
        "residual_vs_fitted": bins,
        "abs_residual_fitted_correlation": round(float(abs_corr), 4),
    }


def _confidence_calibration(frame: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    if "confidence" not in frame.columns:
        return {"bins": [], "ece": 0.0}

    work = frame.dropna(subset=["confidence"]).copy()
    if work.empty:
        return {"bins": [], "ece": 0.0}

    work["confidence"] = work["confidence"].clip(0.0, 1.0)
    work["correct"] = (work["y_true"] == work["y_pred"]).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    work["bin"] = pd.cut(
        work["confidence"],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = float(len(work))
    for bucket, group in work.groupby("bin", observed=False):
        if group.empty or bucket is None:
            continue
        avg_conf = float(group["confidence"].mean())
        accuracy = float(group["correct"].mean())
        weight = len(group) / total if total else 0.0
        ece += weight * abs(accuracy - avg_conf)
        bins.append(
            {
                "range": [round(float(bucket.left), 4), round(float(bucket.right), 4)],
                "count": int(len(group)),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(accuracy, 4),
            }
        )

    return {"bins": bins, "ece": round(float(ece), 4)}


def _residual_bins(frame: pd.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    if len(frame) == 0:
        return []
    work = frame.copy()
    n_unique = int(work["y_pred"].nunique())
    bins = min(max(1, n_bins), max(1, n_unique))
    try:
        work["bin"] = pd.qcut(work["y_pred"], q=bins, duplicates="drop")
    except ValueError:
        return []

    summaries: list[dict[str, Any]] = []
    for bucket, group in work.groupby("bin", observed=False):
        if group.empty or bucket is None:
            continue
        residuals = group["residual"]
        summaries.append(
            {
                "fitted_range": [round(float(bucket.left), 4), round(float(bucket.right), 4)],
                "count": int(len(group)),
                "mean_fitted": round(float(group["y_pred"].mean()), 4),
                "mean_residual": round(float(residuals.mean()), 4),
                "std_residual": round(
                    float(residuals.std(ddof=1)) if len(group) > 1 else 0.0,
                    4,
                ),
            }
        )
    return summaries
