"""Regression metrics for factor score prediction: MAE, R², stratified analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore


@dataclass
class RegressionMetrics:
    mae: float
    r_squared: float
    per_factor_mae: dict[str, float]
    per_factor_r2: dict[str, float]
    factor_names: list[str]
    n_samples: int


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    factor_names: list[str],
) -> RegressionMetrics:
    """
    Compute MAE and R² per factor and overall.
    y_true, y_pred: (n_samples, n_factors) arrays.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    overall_mae = float(mean_absolute_error(y_true, y_pred))
    overall_r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))

    per_factor_mae: dict[str, float] = {}
    per_factor_r2: dict[str, float] = {}

    for i, name in enumerate(factor_names):
        if i >= y_true.shape[1]:
            break
        per_factor_mae[name] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        per_factor_r2[name] = float(r2_score(y_true[:, i], y_pred[:, i]))

    return RegressionMetrics(
        mae=overall_mae,
        r_squared=overall_r2,
        per_factor_mae=per_factor_mae,
        per_factor_r2=per_factor_r2,
        factor_names=factor_names,
        n_samples=len(y_true),
    )


def stratify_regression_by(
    df: pd.DataFrame,
    y_true_cols: list[str],
    y_pred_cols: list[str],
    stratify_col: str,
    factor_names: list[str],
) -> dict[str, RegressionMetrics]:
    """Compute regression metrics stratified by a column (e.g., language, detail_level)."""
    results: dict[str, RegressionMetrics] = {}
    for stratum in df[stratify_col].dropna().unique():
        mask = df[stratify_col] == stratum
        subset = df[mask]
        if len(subset) < 2:
            continue
        y_true = subset[y_true_cols].values
        y_pred = subset[y_pred_cols].values
        results[str(stratum)] = compute_regression_metrics(y_true, y_pred, factor_names)
    return results


def metrics_to_dict(m: RegressionMetrics) -> dict[str, Any]:
    return {
        "mae": m.mae,
        "r_squared": m.r_squared,
        "per_factor_mae": m.per_factor_mae,
        "per_factor_r2": m.per_factor_r2,
        "factor_names": m.factor_names,
        "n_samples": m.n_samples,
    }
