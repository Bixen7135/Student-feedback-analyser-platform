"""Tests for dataset branch management, column_roles, and version deletion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.storage.database import Database
from src.storage.dataset_manager import (
    DatasetManager,
    _detect_initial_column_roles,
    _propagate_column_roles,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def db(tmp_dir: Path) -> Database:
    return Database(tmp_dir / "test.db")


@pytest.fixture()
def mgr(db: Database, tmp_dir: Path) -> DatasetManager:
    return DatasetManager(db=db, datasets_dir=tmp_dir / "datasets")


@pytest.fixture()
def csv_file(tmp_dir: Path) -> Path:
    """A small valid CSV fixture."""
    p = tmp_dir / "sample.csv"
    df = pd.DataFrame({
        "text_feedback": ["great course", "needs improvement", "excellent", "too fast", "perfect"],
        "language": ["ru", "kk", "ru", "kk", "ru"],
        "sentiment_class": ["positive", "negative", "positive", "negative", "positive"],
        "detail_level": ["short", "long", "short", "long", "medium"],
    })
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def dataset_id(mgr: DatasetManager, csv_file: Path) -> str:
    meta = mgr.upload_dataset(csv_file, name="Test Dataset", author="alice")
    return meta.id


# ---------------------------------------------------------------------------
# column_roles helpers
# ---------------------------------------------------------------------------


def test_detect_initial_column_roles_standard():
    df = pd.DataFrame({
        "text_feedback": ["a", "b"],
        "language": ["ru", "kk"],
        "sentiment_class": ["pos", "neg"],
        "detail_level": ["short", "long"],
    })
    roles = _detect_initial_column_roles(df)
    assert roles["text_feedback"] == "text"
    assert roles["language"] == "language"
    assert roles["sentiment_class"] == "sentiment"
    assert roles["detail_level"] == "detail_level"


def test_detect_initial_column_roles_no_standard_text():
    """Falls back to first text candidate found."""
    df = pd.DataFrame({
        "my_col": ["a", "b"],
        "sentiment_class": ["pos", "neg"],
    })
    roles = _detect_initial_column_roles(df)
    # No text candidates matched; sentinel text role absent
    assert "my_col" not in roles or roles.get("my_col") != "text"
    assert roles.get("sentiment_class") == "sentiment"


def test_propagate_column_roles_simple():
    existing = {"text_feedback": "text", "sentiment_class": "sentiment", "other": "language"}
    renames = {"text_feedback": "my_text", "other": "lang_col"}
    result = _propagate_column_roles(existing, renames)
    assert result["my_text"] == "text"
    assert result["lang_col"] == "language"
    assert result["sentiment_class"] == "sentiment"
    assert "text_feedback" not in result
    assert "other" not in result


def test_propagate_column_roles_no_change():
    existing = {"col_a": "text"}
    result = _propagate_column_roles(existing, {"col_b": "col_c"})
    assert result == {"col_a": "text"}  # col_a not renamed


# ---------------------------------------------------------------------------
# Upload creates main branch and column_roles
# ---------------------------------------------------------------------------


def test_upload_creates_main_branch(mgr: DatasetManager, dataset_id: str):
    branches = mgr.list_branches(dataset_id)
    assert len(branches) == 1
    b = branches[0]
    assert b.name == "main"
    assert b.is_default is True
    assert b.is_deleted is False


def test_upload_creates_column_roles(mgr: DatasetManager, dataset_id: str):
    roles = mgr.get_column_roles(dataset_id)
    assert roles.get("text_feedback") == "text"
    assert roles.get("language") == "language"
    assert roles.get("sentiment_class") == "sentiment"
    assert roles.get("detail_level") == "detail_level"


def test_upload_version_has_branch_id(mgr: DatasetManager, dataset_id: str):
    versions = mgr.get_dataset_versions(dataset_id)
    assert len(versions) == 1
    assert versions[0].branch_id is not None


def test_upload_default_branch_id_set(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    assert ds.default_branch_id is not None


# ---------------------------------------------------------------------------
# Branch CRUD
# ---------------------------------------------------------------------------


def test_create_branch_basic(mgr: DatasetManager, dataset_id: str):
    branch = mgr.create_branch(dataset_id, name="experiment", author="bob", description="Testing")
    assert branch.name == "experiment"
    assert branch.is_default is False
    assert branch.dataset_id == dataset_id
    assert branch.author == "bob"
    assert branch.description == "Testing"


def test_create_branch_duplicate_name_raises(mgr: DatasetManager, dataset_id: str):
    mgr.create_branch(dataset_id, name="exp")
    with pytest.raises(ValueError, match="already exists"):
        mgr.create_branch(dataset_id, name="exp")


def test_create_branch_empty_name_raises(mgr: DatasetManager, dataset_id: str):
    with pytest.raises(ValueError, match="empty"):
        mgr.create_branch(dataset_id, name="  ")


def test_create_branch_with_from_version_id(mgr: DatasetManager, dataset_id: str):
    versions = mgr.get_dataset_versions(dataset_id)
    v1 = versions[0]
    branch = mgr.create_branch(dataset_id, name="from-v1", from_version_id=v1.id)
    assert branch.base_version_id == v1.id


def test_create_branch_invalid_from_version_raises(mgr: DatasetManager, dataset_id: str):
    with pytest.raises(ValueError, match="not found"):
        mgr.create_branch(dataset_id, name="bad", from_version_id="nonexistent-uuid")


def test_list_branches(mgr: DatasetManager, dataset_id: str):
    mgr.create_branch(dataset_id, name="a")
    mgr.create_branch(dataset_id, name="b")
    branches = mgr.list_branches(dataset_id)
    names = [b.name for b in branches]
    assert "main" in names
    assert "a" in names
    assert "b" in names


def test_get_branch(mgr: DatasetManager, dataset_id: str):
    created = mgr.create_branch(dataset_id, name="lookup")
    fetched = mgr.get_branch(created.id)
    assert fetched is not None
    assert fetched.name == "lookup"


def test_update_branch_name(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="old-name")
    updated = mgr.update_branch(dataset_id, b.id, name="new-name")
    assert updated.name == "new-name"


def test_update_branch_description(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="mybranch")
    updated = mgr.update_branch(dataset_id, b.id, description="New desc")
    assert updated.description == "New desc"


def test_update_branch_duplicate_name_raises(mgr: DatasetManager, dataset_id: str):
    mgr.create_branch(dataset_id, name="alpha")
    b = mgr.create_branch(dataset_id, name="beta")
    with pytest.raises(ValueError, match="already exists"):
        mgr.update_branch(dataset_id, b.id, name="alpha")


def test_delete_branch(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="to-delete")
    result = mgr.delete_branch(dataset_id, b.id)
    assert result["deleted"] is True
    branches = mgr.list_branches(dataset_id)
    assert not any(br.id == b.id for br in branches)


def test_delete_default_branch_raises(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    with pytest.raises(ValueError, match="default"):
        mgr.delete_branch(dataset_id, ds.default_branch_id)  # type: ignore


def test_set_default_branch(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="new-default")
    mgr.set_default_branch(dataset_id, b.id)
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    assert ds.default_branch_id == b.id
    branches = mgr.list_branches(dataset_id)
    new_default = next((br for br in branches if br.id == b.id), None)
    assert new_default is not None
    assert new_default.is_default is True
    old_main = next((br for br in branches if br.name == "main"), None)
    assert old_main is not None
    assert old_main.is_default is False


# ---------------------------------------------------------------------------
# Version creation on branches
# ---------------------------------------------------------------------------


def test_create_version_on_main_branch(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    df = pd.DataFrame({"text_feedback": ["x"], "language": ["ru"], "sentiment_class": ["pos"], "detail_level": ["short"]})
    ver = mgr.create_version(dataset_id, df, reason="test", branch_id=ds.default_branch_id)
    assert ver.version == 2
    assert ver.branch_id == ds.default_branch_id
    # datasets.current_version should update
    ds_updated = mgr.get_dataset(dataset_id)
    assert ds_updated is not None
    assert ds_updated.current_version == 2


def test_create_version_on_secondary_branch_does_not_update_current(
    mgr: DatasetManager, dataset_id: str
):
    b = mgr.create_branch(dataset_id, name="side")
    df = pd.DataFrame({"text_feedback": ["y"], "language": ["kk"], "sentiment_class": ["neg"], "detail_level": ["long"]})
    ver = mgr.create_version(dataset_id, df, reason="side edit", branch_id=b.id)
    assert ver.branch_id == b.id
    # current_version should still be 1 (main branch)
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    assert ds.current_version == 1


def test_branch_head_version(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    branches = mgr.list_branches(dataset_id)
    main_branch = next((b for b in branches if b.name == "main"), None)
    assert main_branch is not None
    df = pd.DataFrame({"text_feedback": ["z"], "language": ["ru"], "sentiment_class": ["pos"], "detail_level": ["medium"]})
    mgr.create_version(dataset_id, df, branch_id=main_branch.id)
    head = mgr.get_branch_head_version(main_branch.id)
    assert head is not None
    assert head.version == 2


def test_get_dataset_versions_filtered_by_branch(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="filtered")
    # Branch now has a forked version automatically
    branch_versions_before = mgr.get_dataset_versions(dataset_id, branch_id=b.id)
    assert len(branch_versions_before) == 1
    assert branch_versions_before[0].is_fork is True

    # Create another version on the branch
    df = pd.DataFrame({"text_feedback": ["w"], "language": ["ru"], "sentiment_class": ["pos"], "detail_level": ["long"]})
    mgr.create_version(dataset_id, df, branch_id=b.id)

    all_versions = mgr.get_dataset_versions(dataset_id)
    branch_versions = mgr.get_dataset_versions(dataset_id, branch_id=b.id)
    assert len(branch_versions) == 2  # fork + new version
    assert len(all_versions) == 3  # initial + fork + new version
    assert branch_versions[0].branch_id == b.id
    assert branch_versions[1].branch_id == b.id
    # First version on branch should be the fork
    assert branch_versions[1].is_fork is True
    # Second version should not be a fork
    assert branch_versions[0].is_fork is False


# ---------------------------------------------------------------------------
# Column roles propagate through rename
# ---------------------------------------------------------------------------


def test_rename_columns_propagates_column_roles(mgr: DatasetManager, dataset_id: str):
    # Initial: text_feedback → "text"
    initial_roles = mgr.get_column_roles(dataset_id)
    assert initial_roles.get("text_feedback") == "text"

    ver = mgr.rename_columns(dataset_id, renames={"text_feedback": "my_text"})
    new_roles = ver.column_roles
    assert new_roles.get("my_text") == "text"
    assert "text_feedback" not in new_roles
    assert new_roles.get("language") == "language"


def test_rename_columns_updates_stored_column_roles(mgr: DatasetManager, dataset_id: str):
    mgr.rename_columns(dataset_id, renames={"text_feedback": "txt"})
    roles = mgr.get_column_roles(dataset_id)  # current version
    assert roles.get("txt") == "text"
    assert "text_feedback" not in roles


def test_column_roles_by_version(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    v1 = ds.current_version  # has "text_feedback"

    mgr.rename_columns(dataset_id, renames={"text_feedback": "new_text"})
    ds2 = mgr.get_dataset(dataset_id)
    assert ds2 is not None
    v2 = ds2.current_version

    roles_v1 = mgr.get_column_roles(dataset_id, version=v1)
    roles_v2 = mgr.get_column_roles(dataset_id, version=v2)

    assert roles_v1.get("text_feedback") == "text"
    assert roles_v2.get("new_text") == "text"
    assert "text_feedback" not in roles_v2


# ---------------------------------------------------------------------------
# Version metadata update
# ---------------------------------------------------------------------------


def test_update_version_metadata(mgr: DatasetManager, dataset_id: str):
    versions = mgr.get_dataset_versions(dataset_id)
    v = versions[0]
    updated = mgr.update_version_metadata(dataset_id, v.id, reason="Fixed description")
    assert updated.reason == "Fixed description"
    assert updated.id == v.id


def test_update_version_metadata_not_found(mgr: DatasetManager, dataset_id: str):
    with pytest.raises(ValueError, match="not found"):
        mgr.update_version_metadata(dataset_id, "nonexistent-uuid", reason="x")


# ---------------------------------------------------------------------------
# Version deletion
# ---------------------------------------------------------------------------


def test_delete_version_requires_multiple_versions(mgr: DatasetManager, dataset_id: str):
    versions = mgr.get_dataset_versions(dataset_id)
    v = versions[0]
    with pytest.raises(ValueError, match="only version"):
        mgr.delete_version(dataset_id, v.id)


def test_delete_version_success(mgr: DatasetManager, dataset_id: str):
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    df = pd.DataFrame({"text_feedback": ["v2"], "language": ["ru"], "sentiment_class": ["pos"], "detail_level": ["short"]})
    v2 = mgr.create_version(dataset_id, df, reason="v2")

    versions_before = mgr.get_dataset_versions(dataset_id)
    assert len(versions_before) == 2

    # Delete v1 (not head)
    v1 = next(v for v in versions_before if v.version != v2.version)
    result = mgr.delete_version(dataset_id, v1.id)
    assert result["deleted"] is True

    versions_after = mgr.get_dataset_versions(dataset_id)
    assert len(versions_after) == 1
    assert versions_after[0].version == v2.version


def test_delete_version_updates_current_version_when_head_deleted(
    mgr: DatasetManager, dataset_id: str
):
    """Deleting the head of the default branch should update datasets.current_version."""
    ds = mgr.get_dataset(dataset_id)
    assert ds is not None
    df = pd.DataFrame({"text_feedback": ["v2"], "language": ["ru"], "sentiment_class": ["pos"], "detail_level": ["short"]})
    v2 = mgr.create_version(dataset_id, df)

    ds_mid = mgr.get_dataset(dataset_id)
    assert ds_mid is not None
    assert ds_mid.current_version == v2.version

    # Delete head (v2)
    mgr.delete_version(dataset_id, v2.id)

    ds_after = mgr.get_dataset(dataset_id)
    assert ds_after is not None
    assert ds_after.current_version == 1  # rolled back to v1


def test_delete_version_not_found(mgr: DatasetManager, dataset_id: str):
    with pytest.raises(ValueError, match="not found"):
        mgr.delete_version(dataset_id, "nonexistent-uuid")


# ---------------------------------------------------------------------------
# Edit operations on branches
# ---------------------------------------------------------------------------


def test_update_cells_on_branch(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="edit-branch")
    # Branch has no versions yet; _load_branch_head_df falls back to main head data
    ver = mgr.update_cells(
        dataset_id,
        changes=[{"row_idx": 0, "col": "text_feedback", "value": "changed"}],
        reason="cell edit on branch",
        branch_id=b.id,
    )
    assert ver.branch_id == b.id
    # Main branch untouched
    main_branch = next(br for br in mgr.list_branches(dataset_id) if br.name == "main")
    main_head = mgr.get_branch_head_version(main_branch.id)
    assert main_head is not None
    assert main_head.version < ver.version  # new version created on branch


def test_rename_columns_on_secondary_branch(mgr: DatasetManager, dataset_id: str):
    b = mgr.create_branch(dataset_id, name="rename-branch")
    ver = mgr.rename_columns(
        dataset_id,
        renames={"text_feedback": "branch_text"},
        branch_id=b.id,
    )
    assert ver.branch_id == b.id
    assert ver.column_roles.get("branch_text") == "text"
    # Main branch still has original names
    main_roles = mgr.get_column_roles(dataset_id)  # current = main head
    assert main_roles.get("text_feedback") == "text"


# ---------------------------------------------------------------------------
# Migration: old datasets without branches
# ---------------------------------------------------------------------------


def test_get_or_create_main_branch_for_old_dataset(mgr: DatasetManager, dataset_id: str):
    """Simulate a dataset that has no branch_id on versions (pre-Phase7)."""
    # Reset branch_id to NULL on all versions to simulate old data
    mgr.db.execute(
        "UPDATE dataset_versions SET branch_id = NULL WHERE dataset_id = ?", (dataset_id,)
    )
    mgr.db.execute(
        "UPDATE datasets SET default_branch_id = NULL WHERE id = ?", (dataset_id,)
    )
    mgr.db.execute(
        "DELETE FROM dataset_branches WHERE dataset_id = ?", (dataset_id,)
    )
    mgr.db.commit()

    # Calling _get_or_create_main_branch should fix it
    branch = mgr._get_or_create_main_branch(dataset_id, author="migrator")
    assert branch.name == "main"
    assert branch.is_default is True

    # Version should now have branch_id set
    versions = mgr.get_dataset_versions(dataset_id)
    assert all(v.branch_id == branch.id for v in versions)
