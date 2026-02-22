"""Tests for Phase 6 dataset versioning: create_version, update_cells, add_rows, delete_rows."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager


@pytest.fixture
def mgr(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    return DatasetManager(db=db, datasets_dir=tmp_path / "datasets")


def _make_csv(tmp_path: Path, rows: int = 5) -> Path:
    """Write a small CSV and return its path."""
    p = tmp_path / "input.csv"
    df = pd.DataFrame({
        "id": [str(i) for i in range(rows)],
        "name": [f"student_{i}" for i in range(rows)],
        "score": [str(i * 10) for i in range(rows)],
    })
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# create_version
# ---------------------------------------------------------------------------

def test_create_version_increments(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path), name="DS")
    assert ds.current_version == 1

    df_v1 = pd.read_csv(mgr.get_csv_path(ds.id, 1))
    new_df = df_v1.copy()
    new_df.at[0, "name"] = "CHANGED"

    ver = mgr.create_version(ds.id, new_df, reason="test edit")
    assert ver.version == 2
    assert ver.row_count == len(new_df)

    updated_ds = mgr.get_dataset(ds.id)
    assert updated_ds.current_version == 2


def test_create_version_file_saved(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path), name="DS2")
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    ver = mgr.create_version(ds.id, df, reason="overwrite")

    saved_path = Path(ver.storage_path)
    assert saved_path.exists()
    loaded = pd.read_csv(saved_path, dtype=str)
    assert loaded.at[0, "a"] == "x"


def test_create_version_unknown_dataset_raises(mgr):
    with pytest.raises(ValueError, match="not found"):
        mgr.create_version("nonexistent-id", pd.DataFrame({"a": ["1"]}))


# ---------------------------------------------------------------------------
# update_cells
# ---------------------------------------------------------------------------

def test_update_cells_applies_change(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path), name="CellDS")
    ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "NEW_NAME"}], reason="edit name")

    assert ver.version == 2
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    assert df.at[0, "name"] == "NEW_NAME"


def test_update_cells_multiple_changes(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path), name="MultiCell")
    changes = [
        {"row_idx": 0, "col": "score", "value": "999"},
        {"row_idx": 2, "col": "name", "value": "ALICE"},
    ]
    ver = mgr.update_cells(ds.id, changes)
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    assert df.at[0, "score"] == "999"
    assert df.at[2, "name"] == "ALICE"


def test_update_cells_invalid_col_raises(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path), name="BadCol")
    with pytest.raises(ValueError, match="Column not found"):
        mgr.update_cells(ds.id, [{"row_idx": 0, "col": "NONEXISTENT", "value": "x"}])


def test_update_cells_out_of_range_raises(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="OutRange")
    with pytest.raises(ValueError, match="Row index out of range"):
        mgr.update_cells(ds.id, [{"row_idx": 999, "col": "name", "value": "x"}])


# ---------------------------------------------------------------------------
# add_rows
# ---------------------------------------------------------------------------

def test_add_rows_increases_count(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=5), name="AddRows")
    new_rows = [{"id": "100", "name": "extra", "score": "50"}]
    ver = mgr.add_rows(ds.id, new_rows, reason="append")

    assert ver.row_count == 6
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    assert len(df) == 6
    assert df.iloc[-1]["name"] == "extra"


def test_add_rows_fills_missing_cols(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="FillMissing")
    # Provide only 'id', leave 'name' and 'score' unset
    ver = mgr.add_rows(ds.id, [{"id": "99"}])
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    assert len(df) == 4
    assert df.iloc[-1]["id"] == "99"
    assert df.iloc[-1]["name"] == ""


def test_add_rows_to_empty_dataset(mgr, tmp_path):
    ds = mgr.create_empty_dataset("Empty", columns=["a", "b"])
    ver = mgr.add_rows(ds.id, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
    assert ver.row_count == 2


# ---------------------------------------------------------------------------
# delete_rows
# ---------------------------------------------------------------------------

def test_delete_rows_decreases_count(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=5), name="DelRows")
    ver = mgr.delete_rows(ds.id, [0, 2], reason="remove")

    assert ver.row_count == 3
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    assert len(df) == 3
    # Original row 1 should now be at index 0
    assert df.at[0, "id"] == "1"


def test_delete_rows_resets_index(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=4), name="ResetIdx")
    ver = mgr.delete_rows(ds.id, [1])
    df = pd.read_csv(mgr.get_csv_path(ds.id, ver.version), dtype=str, keep_default_na=False)
    # Index should be 0-based after reset
    assert list(df.index) == [0, 1, 2]


def test_delete_rows_out_of_bounds_raises(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="OOB")
    with pytest.raises(ValueError, match="out of range"):
        mgr.delete_rows(ds.id, [100])


def test_delete_all_rows_raises(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=2), name="DelAll")
    with pytest.raises(ValueError, match="empty"):
        mgr.delete_rows(ds.id, [0, 1])


# ---------------------------------------------------------------------------
# Version chain
# ---------------------------------------------------------------------------

def test_version_chain(mgr, tmp_path):
    """Multiple edits produce incrementing versions."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=4), name="Chain")
    v2 = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "A"}])
    v3 = mgr.add_rows(ds.id, [{"id": "X", "name": "B", "score": "0"}])
    v4 = mgr.delete_rows(ds.id, [0])

    assert v2.version == 2
    assert v3.version == 3
    assert v4.version == 4

    history = mgr.get_dataset_versions(ds.id)
    assert len(history) == 4
