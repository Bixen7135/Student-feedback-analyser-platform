"""Unit tests for contradiction monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contradiction.detector import detect_contradictions


def _make_test_data(n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic test dataframe with known contradiction cases."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "survey_id": range(n),
        "language": rng.choice(["ru", "kz", "mixed"], size=n),
        "detail_level": rng.choice(["short", "medium", "long"], size=n),
        "sentiment_class": rng.choice(["positive", "neutral", "negative"], size=n),
    })
    # Force some known contradictions
    df.loc[0, "sentiment_class"] = "negative"
    df.loc[1, "sentiment_class"] = "positive"

    scores = pd.DataFrame({
        "factor1": rng.uniform(0, 1, size=n),
        "factor2": rng.uniform(0, 1, size=n),
        "factor3": rng.uniform(0, 1, size=n),
    })
    # Force row 0 to have very high scores (top decile) — contradiction with negative sentiment
    scores.loc[0] = [0.99, 0.98, 0.99]
    # Force row 1 to have very low scores (bottom decile) — contradiction with positive sentiment
    scores.loc[1] = [0.01, 0.02, 0.01]

    return df, scores


def test_known_type_a_contradiction_flagged():
    df, scores = _make_test_data()
    result = detect_contradictions(df, scores)
    assert result.flags.loc[0, "contradiction_type_a"] == True


def test_known_type_b_contradiction_flagged():
    df, scores = _make_test_data()
    result = detect_contradictions(df, scores)
    assert result.flags.loc[1, "contradiction_type_b"] == True


def test_contradiction_flags_never_alter_predictions():
    """Verify that the detect_contradictions function only reads data, doesn't modify it."""
    df, scores = _make_test_data()
    df_copy = df.copy()
    scores_copy = scores.copy()
    detect_contradictions(df, scores)
    pd.testing.assert_frame_equal(df, df_copy)
    pd.testing.assert_frame_equal(scores, scores_copy)


def test_overall_rate_in_range():
    df, scores = _make_test_data()
    result = detect_contradictions(df, scores)
    assert 0.0 <= result.overall_rate <= 1.0


def test_stratified_rates_by_language():
    df, scores = _make_test_data()
    result = detect_contradictions(df, scores)
    assert len(result.stratified_by_language) > 0
    for rate in result.stratified_by_language.values():
        assert 0.0 <= rate <= 1.0


def test_n_contradictions_consistent():
    df, scores = _make_test_data()
    result = detect_contradictions(df, scores)
    assert result.n_contradictions == int(result.flags["is_contradiction"].sum())
