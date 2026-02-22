"""Unit tests for data splitting."""
from __future__ import annotations

import pytest

from src.splits.splitter import stratified_split, validate_split_no_leakage


def test_split_returns_three_dataframes(preprocessed_df):
    train, val, test = stratified_split(preprocessed_df, seed=42)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_split_total_rows_preserved(preprocessed_df):
    n = len(preprocessed_df)
    train, val, test = stratified_split(preprocessed_df, seed=42)
    assert len(train) + len(val) + len(test) == n


def test_split_ratios_approximately_correct(preprocessed_df):
    n = len(preprocessed_df)
    train, val, test = stratified_split(preprocessed_df, seed=42)
    # Allow 5% tolerance
    assert abs(len(train) / n - 0.80) < 0.05
    assert abs(len(val) / n - 0.10) < 0.05
    assert abs(len(test) / n - 0.10) < 0.05


def test_split_stratification_preserves_distribution(preprocessed_df):
    train, val, test = stratified_split(preprocessed_df, stratify_col="sentiment_class", seed=42)
    orig_dist = preprocessed_df["sentiment_class"].value_counts(normalize=True)
    train_dist = train["sentiment_class"].value_counts(normalize=True)
    for cls in orig_dist.index:
        if cls in train_dist.index:
            # Distribution within 20% relative tolerance (small dataset)
            assert abs(train_dist[cls] - orig_dist[cls]) < 0.20


def test_split_no_survey_id_overlap(preprocessed_df):
    train, val, test = stratified_split(preprocessed_df, seed=42)
    validate_split_no_leakage(train, val, test)  # Should not raise


def test_split_deterministic_with_seed(preprocessed_df):
    train1, val1, test1 = stratified_split(preprocessed_df, seed=42)
    train2, val2, test2 = stratified_split(preprocessed_df, seed=42)
    assert list(train1.index) == list(train2.index)
    assert list(val1.index) == list(val2.index)
    assert list(test1.index) == list(test2.index)
