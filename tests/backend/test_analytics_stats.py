"""Unit tests for analytics statistics helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.analytics.correlations import (
    cramers_v,
    pearson_correlation,
    point_biserial_correlation,
    spearman_correlation,
)
from src.analytics.descriptive import (
    bootstrap_interval,
    categorical_frequency,
    descriptive_summary,
    numeric_summary,
    text_length_stats,
    t_interval_mean,
    wilson_interval,
)
from src.analytics.diagnostics import (
    classification_diagnostics,
    regression_diagnostics,
)


def test_wilson_interval_deterministic() -> None:
    interval = wilson_interval(5, 10)
    assert interval["estimate"] == 0.5
    assert interval["low"] == 0.2366
    assert interval["high"] == 0.7634


def test_t_interval_mean_contains_mean() -> None:
    interval = t_interval_mean([1, 2, 3, 4, 5])
    assert interval["estimate"] == 3.0
    assert interval["low"] < 3.0 < interval["high"]


def test_bootstrap_interval_constant_series() -> None:
    interval = bootstrap_interval([2.0, 2.0, 2.0], metric_fn=lambda arr: float(arr.mean()))
    assert interval["estimate"] == 2.0
    assert interval["low"] == 2.0
    assert interval["high"] == 2.0


def test_numeric_summary_basic() -> None:
    summary = numeric_summary(pd.Series([1, 2, 3, 4]))
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5


def test_categorical_frequency_basic() -> None:
    summary = categorical_frequency(pd.Series(["a", "a", "b", ""]))
    assert summary["count"] == 3
    assert summary["missing"] == 1
    assert summary["levels"]["a"]["count"] == 2


def test_text_length_stats_basic() -> None:
    summary = text_length_stats(pd.Series(["one two", "abc", ""]))
    assert summary["count"] == 2
    assert summary["missing"] == 1
    assert summary["word_length"]["mean"] == 1.5


def test_descriptive_summary_splits_column_types() -> None:
    df = pd.DataFrame(
        {
            "score": ["1", "2", "3"],
            "label": ["positive", "negative", "positive"],
            "text_feedback": ["great course", "needs work", "fine"],
        }
    )
    summary = descriptive_summary(df)
    assert "score" in summary["numeric"]
    assert "label" in summary["categorical"]
    assert "text_feedback" in summary["text"]


def test_pearson_correlation_perfect_positive() -> None:
    corr = pearson_correlation([1, 2, 3], [2, 4, 6])
    assert corr["value"] == 1.0


def test_spearman_correlation_perfect_negative() -> None:
    corr = spearman_correlation([1, 2, 3], [30, 20, 10])
    assert corr["value"] == -1.0


def test_cramers_v_identical_categories() -> None:
    corr = cramers_v(["a", "a", "b", "b"], ["x", "x", "y", "y"])
    assert corr["value"] == 1.0


def test_point_biserial_positive_signal() -> None:
    corr = point_biserial_correlation([0, 0, 1, 1], [1.0, 2.0, 5.0, 6.0])
    assert corr["value"] > 0.0


def test_classification_diagnostics_basic() -> None:
    diagnostics = classification_diagnostics(
        y_true=["a", "a", "b", "b"],
        y_pred=["a", "b", "b", "b"],
        confidences=[0.9, 0.4, 0.8, 0.7],
        labels=["a", "b"],
    )
    assert diagnostics["n_rows"] == 4
    assert diagnostics["accuracy"] == 0.75
    assert diagnostics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert diagnostics["ece"] >= 0.0


def test_regression_diagnostics_basic() -> None:
    diagnostics = regression_diagnostics(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[1.1, 1.9, 3.2, 3.8],
    )
    assert diagnostics["n_rows"] == 4
    assert diagnostics["residual_summary"]["mae"] > 0.0
    assert len(diagnostics["residual_vs_fitted"]) >= 1
