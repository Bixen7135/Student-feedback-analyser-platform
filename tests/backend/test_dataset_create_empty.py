"""Tests for Phase 6 create_empty_dataset."""
from __future__ import annotations

import pytest

from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager


@pytest.fixture
def mgr(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    return DatasetManager(db=db, datasets_dir=tmp_path / "datasets")


def test_create_empty_dataset_has_zero_rows(mgr):
    ds = mgr.create_empty_dataset("EmptyDS", columns=["col_a", "col_b"])
    assert ds.row_count == 0
    assert ds.current_version == 1


def test_create_empty_dataset_has_correct_columns(mgr):
    ds = mgr.create_empty_dataset("ColDS", columns=["x", "y", "z"])
    schema_names = [s.name for s in ds.schema_info]
    assert schema_names == ["x", "y", "z"]


def test_create_empty_dataset_appears_in_list(mgr):
    mgr.create_empty_dataset("ListedDS", columns=["a", "b"])
    datasets, total = mgr.list_datasets()
    assert total == 1
    assert datasets[0].name == "ListedDS"


def test_create_empty_dataset_preview_empty(mgr):
    ds = mgr.create_empty_dataset("PreviewDS", columns=["p", "q"])
    preview = mgr.get_dataset_preview(ds.id)
    assert preview["total_rows"] == 0
    assert preview["columns"] == ["p", "q"]
    assert preview["rows"] == []


def test_create_empty_requires_two_columns(mgr):
    with pytest.raises(ValueError, match="2 columns"):
        mgr.create_empty_dataset("OneCol", columns=["only"])


def test_create_empty_rejects_duplicate_columns(mgr):
    with pytest.raises(ValueError, match="Duplicate"):
        mgr.create_empty_dataset("DupCols", columns=["a", "a", "b"])


def test_create_empty_then_add_rows(mgr):
    ds = mgr.create_empty_dataset("GrowDS", columns=["name", "score"])
    ver = mgr.add_rows(ds.id, [
        {"name": "Alice", "score": "95"},
        {"name": "Bob",   "score": "87"},
    ])
    assert ver.row_count == 2
    assert ver.version == 2

    updated = mgr.get_dataset(ds.id)
    assert updated.row_count == 2


def test_create_empty_with_tags_and_author(mgr):
    ds = mgr.create_empty_dataset("TaggedDS", columns=["a", "b"], tags=["test"], author="ada")
    assert ds.tags == ["test"]
    assert ds.author == "ada"


def test_create_empty_csv_file_exists(mgr, tmp_path):
    ds = mgr.create_empty_dataset("FileDS", columns=["c1", "c2"])
    csv_path = mgr.get_csv_path(ds.id, 1)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "c1" in content
    assert "c2" in content
