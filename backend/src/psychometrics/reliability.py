"""Reliability metrics for psychometric factors: Cronbach's alpha, McDonald's omega."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReliabilityReport:
    cronbach_alpha: dict[str, float]   # factor_name -> alpha
    mcdonald_omega: dict[str, float]   # factor_name -> omega
    per_item_alpha_if_deleted: dict[str, dict[str, float]]  # factor -> item -> alpha_if_deleted


def cronbach_alpha(df: pd.DataFrame, item_cols: list[str]) -> float:
    """
    Compute Cronbach's alpha for a set of items.
    α = (k / (k-1)) * (1 - Σvarᵢ / var_total)
    """
    data = df[item_cols].dropna().astype(float)
    k = len(item_cols)
    if k < 2 or len(data) < 2:
        return float("nan")

    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(np.clip(alpha, -1.0, 1.0))


def alpha_if_item_deleted(df: pd.DataFrame, item_cols: list[str]) -> dict[str, float]:
    """Compute alpha if each item is removed — identifies weak items."""
    result: dict[str, float] = {}
    for item in item_cols:
        remaining = [c for c in item_cols if c != item]
        result[item] = cronbach_alpha(df, remaining)
    return result


def mcdonald_omega(loadings_col: pd.Series) -> float:
    """
    Compute McDonald's omega for a single factor.
    ω = (Σλᵢ)² / [(Σλᵢ)² + Σ(1 - λᵢ²)]
    where λᵢ are the item loadings on this factor.
    """
    lambdas = loadings_col.dropna().values.astype(float)
    if len(lambdas) == 0:
        return float("nan")

    sum_lambda = np.sum(lambdas)
    sum_unique = np.sum(1 - lambdas ** 2)
    denominator = sum_lambda ** 2 + sum_unique
    if denominator == 0:
        return float("nan")

    omega = sum_lambda ** 2 / denominator
    return float(np.clip(omega, 0.0, 1.0))


def compute_reliability(
    df: pd.DataFrame,
    factor_structure_factors: dict[str, list[str]],
    loadings: pd.DataFrame,
) -> ReliabilityReport:
    """Compute reliability metrics for all factors."""
    alphas: dict[str, float] = {}
    omegas: dict[str, float] = {}
    alpha_if_deleted: dict[str, dict[str, float]] = {}

    for factor_name, items in factor_structure_factors.items():
        available = [i for i in items if i in df.columns]
        if not available:
            continue

        alphas[factor_name] = cronbach_alpha(df, available)
        alpha_if_deleted[factor_name] = alpha_if_item_deleted(df, available)

        if factor_name in loadings.columns:
            omegas[factor_name] = mcdonald_omega(loadings[factor_name])
        else:
            omegas[factor_name] = float("nan")

        log.info(
            "reliability_computed",
            factor=factor_name,
            alpha=alphas[factor_name],
            omega=omegas[factor_name],
        )

    return ReliabilityReport(
        cronbach_alpha=alphas,
        mcdonald_omega=omegas,
        per_item_alpha_if_deleted=alpha_if_deleted,
    )
