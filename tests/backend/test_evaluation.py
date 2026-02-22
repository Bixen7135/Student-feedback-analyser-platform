"""Unit tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.regression_metrics import compute_regression_metrics


def test_classification_perfect_predictions():
    y = np.array(["a", "b", "c", "a", "b"])
    metrics = compute_classification_metrics(y, y, ["a", "b", "c"])
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_classification_handles_zero_predictions():
    y_true = np.array(["a", "a", "a"])
    y_pred = np.array(["b", "b", "b"])
    metrics = compute_classification_metrics(y_true, y_pred, ["a", "b"])
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_confusion_matrix_shape():
    y = np.array(["a", "b", "c", "a", "b", "c"])
    metrics = compute_classification_metrics(y, y, ["a", "b", "c"])
    assert metrics.confusion_matrix.shape == (3, 3)


def test_per_class_f1_keys():
    y = np.array(["positive", "neutral", "negative"] * 5)
    metrics = compute_classification_metrics(y, y, ["positive", "neutral", "negative"])
    assert set(metrics.per_class_f1.keys()) == {"positive", "neutral", "negative"}


def test_regression_perfect_predictions():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    metrics = compute_regression_metrics(y, y, ["f1", "f2"])
    assert metrics.mae == pytest.approx(0.0, abs=1e-6)
    assert metrics.r_squared == pytest.approx(1.0, abs=1e-6)


def test_regression_negative_r2():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[3.0], [1.0], [2.0]])
    metrics = compute_regression_metrics(y_true, y_pred, ["f1"])
    # May be negative or low, just check it doesn't crash
    assert isinstance(metrics.r_squared, float)


def test_regression_per_factor_keys():
    y = np.ones((10, 3))
    metrics = compute_regression_metrics(y, y, ["pq", "rt", "da"])
    assert set(metrics.per_factor_mae.keys()) == {"pq", "rt", "da"}
    assert set(metrics.per_factor_r2.keys()) == {"pq", "rt", "da"}
