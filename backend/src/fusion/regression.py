"""
Late fusion regression: survey-only, text-only, and combined models.
Targets are psychometric factor scores (standardized during training).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib  # type: ignore
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse  # type: ignore
from sklearn.linear_model import HuberRegressor  # type: ignore
from sklearn.multioutput import MultiOutputRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.evaluation.regression_metrics import compute_regression_metrics, RegressionMetrics
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FusionResult:
    survey_only: RegressionMetrics
    text_only: RegressionMetrics
    late_fusion: RegressionMetrics
    delta_mae: dict[str, float]   # per factor: fusion_mae - survey_only_mae (negative = improvement)
    delta_r2: dict[str, float]    # per factor: fusion_r2 - survey_only_r2 (positive = improvement)
    factor_names: list[str]
    model_type: str = "huber_regression"


def _as_feature_matrix(X: Any) -> Any:
    """Keep sparse matrices sparse; coerce dense features to numpy arrays."""
    if issparse(X):
        return X
    return np.asarray(X)


def _train_huber_multioutput(
    X_train: Any,
    y_train: np.ndarray,
    seed: int,
    epsilon: float = 1.35,
    max_iter: int = 500,
) -> MultiOutputRegressor:
    """Train a MultiOutputRegressor wrapping HuberRegressor (one per factor)."""
    base = HuberRegressor(epsilon=epsilon, max_iter=max_iter)
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def train_fusion_models(
    train_survey: np.ndarray,       # (n_train, 9) survey item scores
    train_text: Any,                # (n_train, n_features) TF-IDF matrix
    train_targets: np.ndarray,      # (n_train, n_factors) factor scores
    test_survey: np.ndarray,
    test_text: Any,
    test_targets: np.ndarray,
    factor_names: list[str],
    seed: int,
    huber_epsilon: float = 1.35,
    huber_max_iter: int = 500,
) -> FusionResult:
    """
    Train and evaluate three regression models:
    1. Survey-only: uses the 9 survey items as features
    2. Text-only: uses TF-IDF text embeddings
    3. Late fusion: concatenates survey + text embeddings

    Targets are standardized on the training set (zero mean, unit variance).
    """
    train_text_m = _as_feature_matrix(train_text)
    test_text_m = _as_feature_matrix(test_text)

    # Standardize targets
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(train_targets)
    y_test = target_scaler.transform(test_targets)

    # Survey-only
    log.info("fusion_training_survey_only")
    m_survey = _train_huber_multioutput(train_survey, y_train, seed, huber_epsilon, huber_max_iter)
    pred_survey = target_scaler.inverse_transform(m_survey.predict(test_survey))
    metrics_survey = compute_regression_metrics(test_targets, pred_survey, factor_names)

    # Text-only
    log.info("fusion_training_text_only")
    m_text = _train_huber_multioutput(train_text_m, y_train, seed, huber_epsilon, huber_max_iter)
    pred_text = target_scaler.inverse_transform(m_text.predict(test_text_m))
    metrics_text = compute_regression_metrics(test_targets, pred_text, factor_names)

    # Late fusion: concatenate without forcing dense arrays.
    log.info("fusion_training_late_fusion")
    if issparse(train_text_m):
        X_train_fused = hstack([csr_matrix(train_survey), train_text_m], format="csr")
        X_test_fused = hstack([csr_matrix(test_survey), test_text_m], format="csr")
    else:
        X_train_fused = np.hstack([train_survey, train_text_m])
        X_test_fused = np.hstack([test_survey, test_text_m])
    m_fusion = _train_huber_multioutput(X_train_fused, y_train, seed, huber_epsilon, huber_max_iter)
    pred_fusion = target_scaler.inverse_transform(m_fusion.predict(X_test_fused))
    metrics_fusion = compute_regression_metrics(test_targets, pred_fusion, factor_names)

    # Compute deltas
    delta_mae = {
        f: round(metrics_fusion.per_factor_mae[f] - metrics_survey.per_factor_mae[f], 4)
        for f in factor_names if f in metrics_fusion.per_factor_mae
    }
    delta_r2 = {
        f: round(metrics_fusion.per_factor_r2[f] - metrics_survey.per_factor_r2[f], 4)
        for f in factor_names if f in metrics_fusion.per_factor_r2
    }

    log.info(
        "fusion_complete",
        survey_mae=round(metrics_survey.mae, 4),
        text_mae=round(metrics_text.mae, 4),
        fusion_mae=round(metrics_fusion.mae, 4),
        delta_mae=delta_mae,
    )

    return FusionResult(
        survey_only=metrics_survey,
        text_only=metrics_text,
        late_fusion=metrics_fusion,
        delta_mae=delta_mae,
        delta_r2=delta_r2,
        factor_names=factor_names,
    )


def save_fusion_models(
    m_survey: Any,
    m_text: Any,
    m_fusion: Any,
    target_scaler: StandardScaler,
    out_dir: Any,
) -> None:
    """Persist fusion models to disk."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_survey, out_dir / "model_survey_only.joblib")
    joblib.dump(m_text, out_dir / "model_text_only.joblib")
    joblib.dump(m_fusion, out_dir / "model_late_fusion.joblib")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")
