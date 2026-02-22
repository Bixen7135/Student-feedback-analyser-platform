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


# ---------------------------------------------------------------------------
# copy_version
# ---------------------------------------------------------------------------


def test_copy_version_creates_new_metadata(mgr, tmp_path):
    """Copy a version with modified metadata, sharing the same data file."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="CopySource")
    original = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "MODIFIED"}], reason="original version")

    # Copy with new metadata
    copied = mgr.copy_version(
        ds.id,
        original.id,
        new_reason="copied with changes",
        author="copier",
    )

    # New version record
    assert copied.id != original.id
    assert copied.version == original.version + 1
    assert copied.reason == "copied with changes"
    assert copied.author == "copier"
    assert copied.created_at != original.created_at

    # Same data (file, hash, row count)
    assert copied.storage_path == original.storage_path
    assert copied.sha256 == original.sha256
    assert copied.row_count == original.row_count
    assert copied.file_size_bytes == original.file_size_bytes

    # Marked as fork
    assert copied.is_fork is True

    # Data unchanged
    df = pd.read_csv(mgr.get_csv_path(ds.id, copied.version), dtype=str, keep_default_na=False)
    assert df.at[0, "name"] == "MODIFIED"


def test_copy_version_to_different_branch(mgr, tmp_path):
    """Copy a version to a different branch."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="BranchCopy")
    main_ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "EDIT"}], reason="main edit")

    # Create a branch (this also creates a forked version)
    branch = mgr.create_branch(ds.id, name="experiment", from_version_id=main_ver.id, author="tester")

    # Get versions after branch creation to account for the forked version
    versions_before = mgr.get_dataset_versions(ds.id)

    # Copy main version to the new branch with different metadata
    copied = mgr.copy_version(
        ds.id,
        main_ver.id,
        new_reason="experiment baseline",
        author="experimenter",
        branch_id=branch.id,
    )

    assert copied.branch_id == branch.id
    assert copied.reason == "experiment baseline"
    # Should be the next global version after branch creation
    assert copied.version == versions_before[0].version + 1


def test_copy_version_invalid_id_raises(mgr, tmp_path):
    """Copying a non-existent version raises ValueError."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=2), name="BadCopy")
    with pytest.raises(ValueError, match="Version not found"):
        mgr.copy_version(ds.id, "nonexistent-id", "copy")


# ---------------------------------------------------------------------------
# move_version_to_branch
# ---------------------------------------------------------------------------


def test_move_version_to_branch(mgr, tmp_path):
    """Move a version from one branch to another."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="MoveTest")
    main_ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "EDIT"}], reason="main edit")

    # Create a new branch
    branch = mgr.create_branch(ds.id, name="experiment", from_version_id=main_ver.id, author="tester")

    # Add another version to main branch
    new_ver = mgr.add_rows(ds.id, [{"id": "X", "name": "NEW", "score": "5"}], reason="new row on main")

    # Move the new version to the experiment branch
    moved = mgr.move_version_to_branch(ds.id, new_ver.id, branch.id, author="mover")

    assert moved.id == new_ver.id  # Same version record
    assert moved.branch_id == branch.id  # Now on experiment branch

    # Verify it's on the target branch
    branch_versions = mgr.get_dataset_versions(ds.id, branch_id=branch.id)
    assert any(v.id == moved.id for v in branch_versions)


def test_move_version_to_same_branch_returns_unchanged(mgr, tmp_path):
    """Moving a version to its current branch returns the version unchanged."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=2), name="SameBranch")
    ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "X"}])

    # Get the default branch ID from the dataset
    target_branch_id = ds.default_branch_id or ver.branch_id

    # Move to same branch
    result = mgr.move_version_to_branch(ds.id, ver.id, target_branch_id)

    assert result.id == ver.id
    assert result.branch_id == ver.branch_id


def test_move_only_version_on_branch_raises(mgr, tmp_path):
    """Moving the only version on a branch raises ValueError."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=2), name="ForkMove")
    ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "X"}])

    # Create branch - this creates a forked version
    branch = mgr.create_branch(ds.id, name="experiment", from_version_id=ver.id)

    # Get the forked version
    branch_versions = mgr.get_dataset_versions(ds.id, branch_id=branch.id)
    forked = next((v for v in branch_versions if v.is_fork), None)
    assert forked is not None

    # Cannot move the only version on a branch
    with pytest.raises(ValueError, match="Cannot move the only version on a branch"):
        mgr.move_version_to_branch(ds.id, forked.id, ds.default_branch_id)


def test_move_forked_copy_version_allowed_when_source_has_other_versions(mgr, tmp_path):
    """A forked/copy version can move if source branch is not emptied."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="ForkMoveAllowed")

    # Create a branch to move into.
    target_branch = mgr.create_branch(ds.id, name="experiment")

    # Copy on main branch marks the new version as is_fork=True.
    source_head = mgr.get_branch_head_version(ds.default_branch_id)
    assert source_head is not None
    copied = mgr.copy_version(
        ds.id,
        source_head.id,
        new_reason="copy to move",
        branch_id=ds.default_branch_id,
    )
    assert copied.is_fork is True

    moved = mgr.move_version_to_branch(ds.id, copied.id, target_branch.id, author="mover")
    assert moved.id == copied.id
    assert moved.branch_id == target_branch.id


def test_move_version_invalid_branch_raises(mgr, tmp_path):
    """Moving to a non-existent branch raises ValueError."""
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=2), name="BadBranch")
    ver = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "X"}])

    with pytest.raises(ValueError, match="Target branch not found"):
        mgr.move_version_to_branch(ds.id, ver.id, "nonexistent-branch-id")


# ---------------------------------------------------------------------------
# restore / set-default (version actions)
# ---------------------------------------------------------------------------


def test_restore_version_creates_new_latest_on_same_branch(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="RestoreTest")
    v2 = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "A"}], reason="edit A")
    _v3 = mgr.update_cells(ds.id, [{"row_idx": 1, "col": "name", "value": "B"}], reason="edit B")

    restored = mgr.restore_version(ds.id, v2.id, reason="restore to A")
    assert restored.version > v2.version
    assert restored.branch_id == v2.branch_id
    assert restored.reason == "restore to A"
    assert restored.is_fork is False
    assert restored.storage_path == v2.storage_path

    ds_updated = mgr.get_dataset(ds.id)
    assert ds_updated is not None
    assert ds_updated.current_version == restored.version


def test_set_version_as_default_switches_branch_and_makes_selected_current(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="SetDefaultVersionTest")
    main_v2 = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "MAIN"}], reason="main edit")

    branch = mgr.create_branch(ds.id, name="experiment", from_version_id=main_v2.id)
    branch_v2 = mgr.update_cells(
        ds.id,
        [{"row_idx": 1, "col": "name", "value": "EXP"}],
        reason="exp edit",
        branch_id=branch.id,
    )

    versions_before = mgr.get_dataset_versions(ds.id)
    result = mgr.set_version_as_default(ds.id, branch_v2.id)
    versions_after = mgr.get_dataset_versions(ds.id)
    assert len(versions_after) == len(versions_before)
    assert result.branch_id == branch.id

    ds_updated = mgr.get_dataset(ds.id)
    assert ds_updated is not None
    assert ds_updated.default_branch_id == branch.id
    assert ds_updated.current_version == result.version

    # set-default should not force branch head to the selected version
    branch_after = mgr.get_branch(branch.id)
    assert branch_after is not None
    assert branch_after.head_version_id == branch_v2.id


def test_set_version_as_default_does_not_change_head_pointer(mgr, tmp_path):
    ds = mgr.upload_dataset(_make_csv(tmp_path, rows=3), name="SetDefaultNoHeadChange")
    v2 = mgr.update_cells(ds.id, [{"row_idx": 0, "col": "name", "value": "A"}], reason="A")
    v3 = mgr.update_cells(ds.id, [{"row_idx": 1, "col": "name", "value": "B"}], reason="B")

    head_before = mgr.get_branch_head_version(ds.default_branch_id)
    assert head_before is not None
    assert head_before.id == v3.id

    _ = mgr.set_version_as_default(ds.id, v2.id)
    head_after = mgr.get_branch_head_version(ds.default_branch_id)
    assert head_after is not None
    assert head_after.id == v3.id
