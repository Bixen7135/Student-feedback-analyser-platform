"""Unit tests for data ingestion."""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.loader import load_dataset, validate_schema, ITEM_COLS, EXPECTED_COLS


def test_load_dataset_renames_columns(tiny_dataset_path):
    df = load_dataset(tiny_dataset_path)
    assert "survey_id" in df.columns
    assert "text_feedback" in df.columns
    assert "language" in df.columns
    assert "sentiment_class" in df.columns
    assert "detail_level" in df.columns
    for col in ITEM_COLS:
        assert col in df.columns


def test_load_dataset_item_types(tiny_dataset_path):
    df = load_dataset(tiny_dataset_path)
    for col in ITEM_COLS:
        assert df[col].dtype.name in ("Int64", "int64", "float64"), f"{col} has wrong dtype"


def test_load_dataset_item_range(tiny_dataset_path):
    df = load_dataset(tiny_dataset_path)
    for col in ITEM_COLS:
        valid = df[col].dropna()
        assert (valid >= 0).all() and (valid <= 10).all(), f"{col} has out-of-range values"


def test_load_dataset_label_values(tiny_dataset_path):
    df = load_dataset(tiny_dataset_path)
    assert set(df["language"].unique()).issubset({"ru", "kz", "mixed"})
    assert set(df["sentiment_class"].unique()).issubset({"positive", "neutral", "negative"})
    assert set(df["detail_level"].unique()).issubset({"short", "medium", "long"})


def test_validate_schema_rejects_missing_column(tiny_dataset_path):
    df = load_dataset(tiny_dataset_path)
    df_bad = df.drop(columns=["item_1"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df_bad)


def test_create_snapshot_deterministic_hash(tiny_dataset_path, tmp_path):
    from src.ingest.snapshot import create_snapshot
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    h1 = create_snapshot(tiny_dataset_path, run_dir)
    h2 = create_snapshot(tiny_dataset_path, run_dir)
    assert h1 == h2


def test_snapshot_copies_file(tiny_dataset_path, tmp_path):
    from src.ingest.snapshot import create_snapshot
    run_dir = tmp_path / "run_snap"
    run_dir.mkdir()
    create_snapshot(tiny_dataset_path, run_dir)
    assert (run_dir / "raw" / tiny_dataset_path.name).exists()
