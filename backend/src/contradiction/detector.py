"""
Contradiction monitoring — deterministic flags for sentiment vs factor score contradictions.
Monitoring ONLY. Never alters predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ContradictionResult:
    flags: pd.DataFrame                          # DataFrame with boolean contradiction columns
    overall_rate: float                          # fraction of contradictions in dataset
    by_contradiction_type: dict[str, float]     # rate per type
    stratified_by_language: dict[str, float]    # rate per language
    stratified_by_length: dict[str, float]      # rate per detail_level
    n_total: int
    n_contradictions: int
    note: str = "Contradiction flags are for monitoring only. Predictions are not altered."


def detect_contradictions(
    df: pd.DataFrame,
    factor_scores: pd.DataFrame,
    sentiment_col: str = "sentiment_class",
    strong_negative_class: str = "negative",
    strong_positive_class: str = "positive",
    high_percentile: float = 90.0,
    low_percentile: float = 10.0,
) -> ContradictionResult:
    """
    Flag rows where:
    - Type A: strong_negative sentiment AND top-decile factor score (any factor)
    - Type B: strong_positive sentiment AND bottom-decile factor score (any factor)

    These are contradictions that may indicate data quality issues or
    complex feedback patterns. Predictions are NEVER altered.
    """
    # Align indices
    aligned_df = df.copy()
    scores = factor_scores.copy()

    if len(aligned_df) != len(scores):
        # Try to align on index
        common_idx = aligned_df.index.intersection(scores.index)
        aligned_df = aligned_df.loc[common_idx]
        scores = scores.loc[common_idx]

    factor_cols = list(scores.columns)

    # Compute percentile thresholds from the scores (one threshold per factor)
    high_thresholds = {f: float(np.percentile(scores[f].dropna(), high_percentile)) for f in factor_cols}
    low_thresholds = {f: float(np.percentile(scores[f].dropna(), low_percentile)) for f in factor_cols}

    # Flag: high factor score (ANY factor above high percentile)
    is_high = scores.apply(lambda col: col >= high_thresholds[col.name], axis=0).any(axis=1)
    # Flag: low factor score (ANY factor below low percentile)
    is_low = scores.apply(lambda col: col <= low_thresholds[col.name], axis=0).any(axis=1)

    is_strong_negative = aligned_df[sentiment_col] == strong_negative_class
    is_strong_positive = aligned_df[sentiment_col] == strong_positive_class

    # Contradiction flags
    flag_a = is_strong_negative & is_high   # Negative sentiment + high scores
    flag_b = is_strong_positive & is_low    # Positive sentiment + low scores

    flags = pd.DataFrame({
        "contradiction_type_a": flag_a,
        "contradiction_type_b": flag_b,
        "is_contradiction": flag_a | flag_b,
    }, index=aligned_df.index)

    n_total = len(flags)
    n_a = int(flag_a.sum())
    n_b = int(flag_b.sum())
    n_any = int(flags["is_contradiction"].sum())
    overall_rate = n_any / n_total if n_total > 0 else 0.0

    # Stratified rates
    strat_lang = _stratified_rate(flags, aligned_df, "language")
    strat_length = _stratified_rate(flags, aligned_df, "detail_level")

    log.info(
        "contradictions_detected",
        n_total=n_total,
        n_type_a=n_a,
        n_type_b=n_b,
        overall_rate=round(overall_rate, 4),
    )

    return ContradictionResult(
        flags=flags,
        overall_rate=overall_rate,
        by_contradiction_type={
            "type_a_negative_high_score": n_a / n_total if n_total > 0 else 0.0,
            "type_b_positive_low_score": n_b / n_total if n_total > 0 else 0.0,
        },
        stratified_by_language=strat_lang,
        stratified_by_length=strat_length,
        n_total=n_total,
        n_contradictions=n_any,
    )


def _stratified_rate(
    flags: pd.DataFrame,
    df: pd.DataFrame,
    col: str,
) -> dict[str, float]:
    if col not in df.columns:
        return {}
    result: dict[str, float] = {}
    for val in df[col].dropna().unique():
        mask = df[col] == val
        subset_flags = flags.loc[mask, "is_contradiction"]
        result[str(val)] = float(subset_flags.mean()) if len(subset_flags) > 0 else 0.0
    return result
