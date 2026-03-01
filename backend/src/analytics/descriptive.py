"""Descriptive statistics and confidence intervals."""
from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

_Z_95 = 1.959963984540054
_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Return a Wilson score interval for a binomial proportion."""
    successes = int(max(0, successes))
    total = int(max(0, total))
    if total == 0:
        return {
            "low": 0.0,
            "high": 0.0,
            "estimate": 0.0,
            "n": 0,
            "method": "wilson",
        }

    z = _z_value(confidence)
    p_hat = successes / total
    z_sq = z * z
    denominator = 1 + z_sq / total
    center = (p_hat + z_sq / (2 * total)) / denominator
    margin = (
        z
        * sqrt((p_hat * (1 - p_hat) + z_sq / (4 * total)) / total)
        / denominator
    )
    return {
        "low": round(max(0.0, center - margin), 4),
        "high": round(min(1.0, center + margin), 4),
        "estimate": round(p_hat, 4),
        "n": total,
        "method": "wilson",
    }


def t_interval_mean(
    values: pd.Series | np.ndarray | list[float],
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Return a t-based confidence interval for the mean when n > 1."""
    arr = _to_numeric_array(values)
    n = int(arr.size)
    if n == 0:
        return {
            "low": 0.0,
            "high": 0.0,
            "estimate": 0.0,
            "n": 0,
            "method": "none",
        }
    mean = float(arr.mean())
    if n == 1:
        return {
            "low": round(mean, 4),
            "high": round(mean, 4),
            "estimate": round(mean, 4),
            "n": 1,
            "method": "degenerate",
        }

    std = float(arr.std(ddof=1))
    if std == 0.0:
        return {
            "low": round(mean, 4),
            "high": round(mean, 4),
            "estimate": round(mean, 4),
            "n": n,
            "method": "t",
        }

    t_critical = _t_critical_value(n - 1, confidence)
    margin = t_critical * (std / sqrt(n))
    return {
        "low": round(mean - margin, 4),
        "high": round(mean + margin, 4),
        "estimate": round(mean, 4),
        "n": n,
        "method": "t",
    }


def bootstrap_interval(
    values: pd.Series | np.ndarray | list[float],
    metric_fn: Callable[[np.ndarray], float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Return a percentile bootstrap interval for an arbitrary metric."""
    arr = _to_numeric_array(values)
    n = int(arr.size)
    if n == 0:
        return {
            "low": 0.0,
            "high": 0.0,
            "estimate": 0.0,
            "n": 0,
            "method": "bootstrap",
        }

    metric = float(metric_fn(arr))
    if n == 1:
        return {
            "low": round(metric, 4),
            "high": round(metric, 4),
            "estimate": round(metric, 4),
            "n": 1,
            "method": "bootstrap",
        }

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        draws[i] = float(metric_fn(sample))

    alpha = 1.0 - confidence
    low_q = alpha / 2
    high_q = 1 - alpha / 2
    low, high = np.quantile(draws, [low_q, high_q])
    return {
        "low": round(float(low), 4),
        "high": round(float(high), 4),
        "estimate": round(metric, 4),
        "n": n,
        "method": "bootstrap",
    }


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    """Summarize a numeric series with location, spread, and CI."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "missing": int(series.isna().sum()),
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "max": 0.0,
            "confidence_interval": t_interval_mean([]),
        }

    q25, median, q75 = numeric.quantile([0.25, 0.5, 0.75]).tolist()
    return {
        "count": int(numeric.count()),
        "missing": int(series.shape[0] - numeric.count()),
        "mean": round(float(numeric.mean()), 4),
        "std": round(float(numeric.std(ddof=1)) if numeric.count() > 1 else 0.0, 4),
        "min": round(float(numeric.min()), 4),
        "q25": round(float(q25), 4),
        "median": round(float(median), 4),
        "q75": round(float(q75), 4),
        "max": round(float(numeric.max()), 4),
        "confidence_interval": t_interval_mean(numeric.to_numpy()),
    }


def categorical_frequency(series: pd.Series, max_levels: int = 25) -> dict[str, Any]:
    """Summarize a categorical series with counts, proportions, and Wilson CIs."""
    text = series.fillna("").astype(str)
    non_empty = text[text.str.strip() != ""]
    total = int(non_empty.shape[0])
    value_counts = non_empty.value_counts()
    levels: dict[str, Any] = {}
    for label, count in value_counts.head(max_levels).items():
        count_int = int(count)
        levels[str(label)] = {
            "count": count_int,
            "proportion": round(count_int / total, 4) if total else 0.0,
            "confidence_interval": wilson_interval(count_int, total),
        }

    top = str(value_counts.index[0]) if not value_counts.empty else None
    return {
        "count": total,
        "missing": int(series.shape[0] - total),
        "n_unique": int(value_counts.shape[0]),
        "top": top,
        "levels": levels,
    }


def text_length_stats(series: pd.Series) -> dict[str, Any]:
    """Summarize text columns by character and token length."""
    text = series.fillna("").astype(str)
    non_empty = text[text.str.strip() != ""]
    if non_empty.empty:
        empty_summary = numeric_summary(pd.Series(dtype=float))
        return {
            "count": 0,
            "missing": int(series.shape[0]),
            "char_length": empty_summary,
            "word_length": empty_summary,
        }

    char_lengths = non_empty.str.len()
    word_lengths = non_empty.str.split().str.len()
    return {
        "count": int(non_empty.shape[0]),
        "missing": int(series.shape[0] - non_empty.shape[0]),
        "char_length": numeric_summary(char_lengths),
        "word_length": numeric_summary(word_lengths),
    }


def descriptive_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build descriptive statistics across numeric, categorical, and text columns."""
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    text: dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_empty = series.fillna("").astype(str).str.strip()
        is_numeric = bool(non_empty.eq("").all()) or numeric_series[non_empty != ""].notna().all()

        if is_numeric:
            numeric[str(col)] = numeric_summary(series)
            continue

        categorical[str(col)] = categorical_frequency(series)
        text[str(col)] = text_length_stats(series)

    return {
        "n_rows": int(len(df)),
        "numeric": numeric,
        "categorical": categorical,
        "text": text,
    }


def _to_numeric_array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
    return arr.astype(float, copy=False)


def _z_value(confidence: float) -> float:
    if abs(confidence - 0.95) < 1e-9:
        return _Z_95
    # Fallback to the common two-sided 95% constant when other levels are requested.
    return _Z_95


def _t_critical_value(df: int, confidence: float) -> float:
    if abs(confidence - 0.95) >= 1e-9:
        return _Z_95
    if df <= 0:
        return _Z_95
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    return _Z_95
