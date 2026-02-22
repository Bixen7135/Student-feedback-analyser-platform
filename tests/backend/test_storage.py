"""Tests for SQLite storage layer and DatasetManager."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager, DatasetValidationError


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def db(tmp_dir):
    return Database(tmp_dir / "test.db")


@pytest.fixture
def mgr(db, tmp_dir):
    return DatasetManager(db, tmp_dir / "datasets")


@pytest.fixture
def sample_csv(tmp_dir) -> Path:
    """Create a small valid CSV file."""
    path = tmp_dir / "sample.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "score", "text"])
        for i in range(10):
            writer.writerow([i, f"student_{i}", i * 10, f"feedback text {i}"])
    return path


@pytest.fixture
def empty_csv(tmp_dir) -> Path:
    path = tmp_dir / "empty.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
    return path


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------


class TestDatabase:
    def test_schema_created(self, db):
        """Tables should be created on init."""
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {r["name"] for r in tables}
        assert "datasets" in names
        assert "dataset_versions" in names
        assert "models" in names
        assert "analysis_runs" in names
        assert "saved_filters" in names

    def test_wal_mode(self, db):
        """Database should use WAL journal mode."""
        row = db.fetchone("PRAGMA journal_mode")
        assert row[0] == "wal"

    def test_foreign_keys_enabled(self, db):
        """Foreign keys should be enforced."""
        row = db.fetchone("PRAGMA foreign_keys")
        assert row[0] == 1

    def test_execute_and_fetch(self, db):
        """Basic execute and fetchall should work."""
        db.execute(
            "INSERT INTO datasets (id, name, created_at) VALUES (?, ?, ?)",
            ("test-1", "Test", "2026-01-01T00:00:00Z"),
        )
        db.commit()
        rows = db.fetchall("SELECT * FROM datasets WHERE id = ?", ("test-1",))
        assert len(rows) == 1
        assert rows[0]["name"] == "Test"


# ---------------------------------------------------------------------------
# DatasetManager tests
# ---------------------------------------------------------------------------


class TestDatasetManager:
    def test_upload_creates_record(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS", author="tester")
        assert meta.name == "Test DS"
        assert meta.author == "tester"
        assert meta.row_count == 10
        assert meta.current_version == 1
        assert meta.status == "active"
        assert len(meta.schema_info) == 4
        assert meta.sha256 != ""

    def test_upload_stores_file(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        stored = mgr.get_csv_path(meta.id)
        assert stored.exists()

    def test_upload_with_tags(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Tagged", tags=["survey", "2026"])
        assert meta.tags == ["survey", "2026"]

    def test_validate_empty_csv_raises(self, mgr, empty_csv):
        with pytest.raises(DatasetValidationError, match="empty"):
            mgr.validate_csv(empty_csv)

    def test_validate_nonexistent_file_raises(self, mgr, tmp_dir):
        with pytest.raises(DatasetValidationError, match="Cannot read"):
            mgr.validate_csv(tmp_dir / "nonexistent.csv")

    def test_list_datasets_empty(self, mgr):
        datasets, total = mgr.list_datasets()
        assert datasets == []
        assert total == 0

    def test_list_datasets_returns_uploaded(self, mgr, sample_csv):
        mgr.upload_dataset(sample_csv, name="DS A")
        mgr.upload_dataset(sample_csv, name="DS B")
        datasets, total = mgr.list_datasets()
        assert total == 2
        assert len(datasets) == 2

    def test_list_datasets_search(self, mgr, sample_csv):
        mgr.upload_dataset(sample_csv, name="Alpha Dataset")
        mgr.upload_dataset(sample_csv, name="Beta Dataset")
        datasets, total = mgr.list_datasets(search="Alpha")
        assert total == 1
        assert datasets[0].name == "Alpha Dataset"

    def test_list_datasets_pagination(self, mgr, sample_csv):
        for i in range(5):
            mgr.upload_dataset(sample_csv, name=f"DS {i}")
        datasets, total = mgr.list_datasets(page=1, per_page=2)
        assert total == 5
        assert len(datasets) == 2

    def test_get_dataset(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        fetched = mgr.get_dataset(meta.id)
        assert fetched is not None
        assert fetched.id == meta.id
        assert fetched.name == "Test DS"

    def test_get_dataset_not_found(self, mgr):
        assert mgr.get_dataset("nonexistent") is None

    def test_get_dataset_preview(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        preview = mgr.get_dataset_preview(meta.id, offset=0, limit=5)
        assert preview["total_rows"] == 10
        assert len(preview["rows"]) == 5
        assert preview["columns"] == ["id", "name", "score", "text"]

    def test_get_dataset_preview_pagination(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        p1 = mgr.get_dataset_preview(meta.id, offset=0, limit=3)
        p2 = mgr.get_dataset_preview(meta.id, offset=3, limit=3)
        assert p1["rows"] != p2["rows"]

    def test_update_metadata(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Original")
        updated = mgr.update_metadata(meta.id, name="Renamed")
        assert updated is not None
        assert updated.name == "Renamed"

    def test_update_metadata_tags(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        updated = mgr.update_metadata(meta.id, tags=["new_tag"])
        assert updated is not None
        assert updated.tags == ["new_tag"]

    def test_delete_dataset_soft(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        result = mgr.delete_dataset(meta.id)
        assert result["deleted"] is True
        # Should not appear in active list
        datasets, total = mgr.list_datasets()
        assert total == 0

    def test_delete_dataset_not_found(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.delete_dataset("nonexistent")

    def test_get_dependencies_empty(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        deps = mgr.get_dependencies(meta.id)
        assert deps["models"] == 0
        assert deps["analyses"] == 0

    def test_get_dataset_versions(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        versions = mgr.get_dataset_versions(meta.id)
        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].reason == "initial upload"

    def test_get_dataframe(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        df = mgr.get_dataframe(meta.id)
        assert len(df) == 10
        # "text" column is normalised to "text_feedback" via column_roles
        assert list(df.columns) == ["id", "name", "score", "text_feedback"]

    def test_get_schema(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Test DS")
        schema = mgr.get_dataset_schema(meta.id)
        assert len(schema) == 4
        col_names = [c.name for c in schema]
        assert "id" in col_names
        assert "text" in col_names


class TestDatasetSubset:
    def test_create_subset_column_equals(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Full DS")
        subset = mgr.create_subset(
            meta.id,
            filter_config={"column_equals": {"name": "student_0"}},
            name="Subset",
        )
        assert subset.row_count == 1
        assert "subset" in subset.tags

    def test_create_subset_column_in(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Full DS")
        subset = mgr.create_subset(
            meta.id,
            filter_config={"column_in": {"name": ["student_0", "student_1", "student_2"]}},
            name="Subset",
        )
        assert subset.row_count == 3

    def test_create_subset_column_contains(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Full DS")
        subset = mgr.create_subset(
            meta.id,
            filter_config={"column_contains": {"text": "text 5"}},
            name="Subset",
        )
        assert subset.row_count == 1

    def test_create_subset_empty_raises(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Full DS")
        with pytest.raises(ValueError, match="empty"):
            mgr.create_subset(
                meta.id,
                filter_config={"column_equals": {"name": "nonexistent_student"}},
                name="Empty Subset",
            )

    def test_create_subset_row_indices(self, mgr, sample_csv):
        meta = mgr.upload_dataset(sample_csv, name="Full DS")
        subset = mgr.create_subset(
            meta.id,
            filter_config={"row_indices": [0, 2, 4]},
            name="Selected Rows",
        )
        assert subset.row_count == 3
