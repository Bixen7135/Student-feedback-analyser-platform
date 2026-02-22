"""Dataset management — upload, validate, CRUD, subset creation, branches."""
from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

from src.ingest.loader import COLUMN_MAP
from src.storage.database import Database
from src.storage.models import ColumnSchema, DatasetBranch, DatasetMeta, DatasetVersion
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column-role detection constants (mirrors training/runner.py without circular dep)
# ---------------------------------------------------------------------------

_TEXT_COL_CANDIDATES = [
    "text_processed",
    "text_feedback",
    "text",
    "feedback",
    "comment",
    "review",
    "response",
]

_TASK_ROLE_COLS: dict[str, str] = {
    "language": "language",
    "sentiment": "sentiment_class",
    "detail_level": "detail_level",
}

# Reverse: role → standard column name used throughout the pipeline.
_ROLE_TO_STANDARD: dict[str, str] = {
    "text": "text_feedback",
    "sentiment": "sentiment_class",
    "language": "language",
    "detail_level": "detail_level",
}

# Fallback aliases used when version column_roles are missing/incomplete.
# Kept intentionally conservative to avoid accidental wrong mappings.
_STANDARD_COL_ALIASES: dict[str, set[str]] = {
    "text_feedback": {
        "textfeedback",
        "text",
        "feedback",
        "comment",
        "review",
        "response",
    },
    "sentiment_class": {
        "sentimentclass",
        "sentiment",
        "\u0442\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u044c\u043a\u043b\u0430\u0441\u0441",
        "\u043a\u043b\u0430\u0441\u0441\u0442\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438",
    },
    "language": {
        "language",
        "\u044f\u0437\u044b\u043a",
    },
    "detail_level": {
        "detaillevel",
        "length",
        "\u0434\u043b\u0438\u043d\u0430",
    },
}


def _compact_col_name(name: str) -> str:
    """Normalize a column name for fuzzy alias matching."""
    return re.sub(r"[\W_]+", "", str(name).strip().lower(), flags=re.UNICODE)


def _fallback_standard_renames(df: pd.DataFrame) -> dict[str, str]:
    """Infer safe renames to canonical pipeline column names by known aliases."""
    renames: dict[str, str] = {}
    by_compact: dict[str, list[str]] = {}
    for col in df.columns:
        key = _compact_col_name(col)
        by_compact.setdefault(key, []).append(col)

    for standard, aliases in _STANDARD_COL_ALIASES.items():
        if standard in df.columns:
            continue
        candidates: list[str] = []
        for alias in aliases:
            candidates.extend(by_compact.get(alias, []))
        # Avoid ambiguous remaps if multiple source columns match.
        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) == 1:
            renames[unique_candidates[0]] = standard
    return renames


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_schema(df: pd.DataFrame) -> list[ColumnSchema]:
    """Infer column schema from a DataFrame."""
    cols: list[ColumnSchema] = []
    for col in df.columns:
        sample = df[col].dropna().head(5).astype(str).tolist()
        cols.append(ColumnSchema(
            name=col,
            dtype=str(df[col].dtype),
            n_unique=int(df[col].nunique()),
            n_null=int(df[col].isna().sum()),
            sample_values=sample,
        ))
    return cols


def _detect_initial_column_roles(df: pd.DataFrame) -> dict[str, str]:
    """Detect initial column roles from column names."""
    roles: dict[str, str] = {}
    cols = set(df.columns)
    for cand in _TEXT_COL_CANDIDATES:
        if cand in cols:
            roles[cand] = "text"
            break
    for role, col in _TASK_ROLE_COLS.items():
        if col in cols:
            roles[col] = role
    return roles


def _propagate_column_roles(
    existing_roles: dict[str, str], renames: dict[str, str]
) -> dict[str, str]:
    """Apply column renames to an existing column_roles mapping."""
    updated: dict[str, str] = {}
    for col, role in existing_roles.items():
        new_col = renames.get(col, col)
        updated[new_col] = role
    return updated


class DatasetValidationError(Exception):
    """Raised when CSV validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class DatasetManager:
    """Manages dataset storage, metadata, branches, and versions."""

    def __init__(self, db: Database, datasets_dir: Path) -> None:
        self.db = db
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_csv(self, path: Path) -> tuple[pd.DataFrame, list[str]]:
        """Validate a CSV file. Returns (df, warnings). Raises DatasetValidationError on critical errors."""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            df = pd.read_csv(path, encoding="utf-8", dtype=str)
        except Exception as e:
            raise DatasetValidationError([f"Cannot read CSV: {e}"])

        if df.empty:
            raise DatasetValidationError(["CSV file is empty"])

        if len(df.columns) < 2:
            errors.append(f"CSV has only {len(df.columns)} column(s)")

        # Check for duplicate column names
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            errors.append(f"Duplicate column names: {dupes}")

        if errors:
            raise DatasetValidationError(errors)

        # Warnings for common issues
        if df.isna().all(axis=1).any():
            n_empty = int(df.isna().all(axis=1).sum())
            warnings.append(f"{n_empty} completely empty row(s)")

        return df, warnings

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_dataset(
        self,
        file_path: Path,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        author: str = "",
    ) -> DatasetMeta:
        """Upload and register a new dataset from a CSV file."""
        tags = tags or []

        # Validate
        df, warnings = self.validate_csv(file_path)
        if warnings:
            log.info("dataset_upload_warnings", warnings=warnings)

        # Compute metadata
        dataset_id = str(uuid.uuid4())
        sha = _sha256(file_path)
        file_size = file_path.stat().st_size
        df_for_schema = df.copy()
        df_for_schema.columns = [c.strip() for c in df_for_schema.columns]
        df_for_schema = df_for_schema.rename(columns=COLUMN_MAP)
        schema_info = _infer_schema(df_for_schema)
        column_roles = _detect_initial_column_roles(df_for_schema)
        now = _utcnow()

        # Create "main" branch first (we need its ID for the version record)
        branch_id = str(uuid.uuid4())

        # Store file
        version = 1
        version_dir = self.datasets_dir / dataset_id / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / "data.csv"
        shutil.copy2(file_path, dest)

        # Write version metadata snapshot
        version_id = str(uuid.uuid4())
        version_meta = {
            "id": version_id,
            "dataset_id": dataset_id,
            "version": version,
            "created_at": now,
            "author": author,
            "reason": "initial upload",
            "sha256": sha,
            "row_count": len(df),
            "file_size_bytes": file_size,
        }
        (version_dir / "metadata.json").write_bytes(
            orjson.dumps(version_meta, option=orjson.OPT_INDENT_2)
        )

        # Insert into DB
        schema_json = orjson.dumps([s.model_dump() for s in schema_info]).decode()
        tags_json = orjson.dumps(tags).decode()
        column_roles_json = orjson.dumps(column_roles).decode()

        self.db.execute(
            """INSERT INTO datasets
            (id, name, description, tags, author, created_at, current_version,
             schema_info, row_count, file_size_bytes, sha256, status, default_branch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, name, description, tags_json, author, now, version,
             schema_json, len(df), file_size, sha, "active", branch_id),
        )
        self.db.execute(
            """INSERT INTO dataset_branches
            (id, dataset_id, name, description, base_version_id, head_version_id, author, created_at, is_default, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (branch_id, dataset_id, "main", "", None, version_id, author, now, 1, 0),
        )
        self.db.execute(
            """INSERT INTO dataset_versions
            (id, dataset_id, version, created_at, author, reason, sha256,
             row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (version_id, dataset_id, version, now, author, "initial upload",
             sha, len(df), file_size, str(version_dir / "data.csv"), branch_id,
             column_roles_json, 0),
        )
        self.db.commit()

        meta = DatasetMeta(
            id=dataset_id,
            name=name,
            description=description,
            tags=tags,
            author=author,
            created_at=now,
            current_version=version,
            schema_info=schema_info,
            row_count=len(df),
            file_size_bytes=file_size,
            sha256=sha,
            default_branch_id=branch_id,
        )
        log.info("dataset_uploaded", id=dataset_id, name=name, rows=len(df))
        return meta

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_datasets(
        self,
        search: str | None = None,
        tags: list[str] | None = None,
        sort: str = "created_at",
        order: str = "desc",
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[DatasetMeta], int]:
        """List datasets with search/filter/pagination. Returns (datasets, total_count)."""
        conditions = ["status = 'active'"]
        params: list[Any] = []

        if search:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        where = " AND ".join(conditions)

        valid_sorts = {"created_at", "name", "row_count", "file_size_bytes"}
        if sort not in valid_sorts:
            sort = "created_at"
        order_dir = "DESC" if order.lower() == "desc" else "ASC"

        count_row = self.db.fetchone(f"SELECT COUNT(*) as cnt FROM datasets WHERE {where}", tuple(params))
        total = count_row["cnt"] if count_row else 0

        offset = (page - 1) * per_page
        rows = self.db.fetchall(
            f"SELECT * FROM datasets WHERE {where} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?",
            (*params, per_page, offset),
        )

        datasets = [self._row_to_dataset(r) for r in rows]
        return datasets, total

    def get_dataset(self, dataset_id: str) -> DatasetMeta | None:
        """Get a single dataset by ID."""
        row = self.db.fetchone("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            return None
        return self._row_to_dataset(row)

    def get_dataset_preview(
        self,
        dataset_id: str,
        version: int | None = None,
        branch_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get paginated rows from a dataset version."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        if version is not None:
            file_path = self._get_version_file(dataset_id, version)
            v = version
        elif branch_id is not None:
            head = self.get_branch_head_version(branch_id)
            if head is None:
                # Fallback for old branches without forked versions (backward compatibility)
                branch = self.get_branch(branch_id)
                if branch and branch.base_version_id:
                    base_row = self.db.fetchone(
                        "SELECT * FROM dataset_versions WHERE id = ?",
                        (branch.base_version_id,),
                    )
                    if base_row:
                        file_path = Path(base_row["storage_path"])
                        v = base_row["version"]
                    else:
                        raise ValueError(f"Base version not found: {branch.base_version_id}")
                else:
                    raise ValueError(f"No versions found on branch {branch_id}")
            else:
                v = head.version
                file_path = self._get_version_file(dataset_id, v)
        else:
            v = ds.current_version
            file_path = self._get_version_file(dataset_id, v)

        df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        total_rows = len(df)
        page = df.iloc[offset: offset + limit]

        return {
            "dataset_id": dataset_id,
            "version": v,
            "columns": list(df.columns),
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "rows": page.fillna("").values.tolist(),
        }

    def get_dataset_schema(self, dataset_id: str) -> list[ColumnSchema]:
        """Get the schema info for a dataset."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        return ds.schema_info

    def get_dataset_versions(
        self,
        dataset_id: str,
        branch_id: str | None = None,
    ) -> list[DatasetVersion]:
        """Get all non-deleted versions of a dataset, optionally filtered by branch."""
        if branch_id is not None:
            branch = self.get_branch(branch_id)
            head_version_id = branch.head_version_id if branch else None
            rows = self.db.fetchall(
                """SELECT * FROM dataset_versions
                   WHERE dataset_id = ? AND branch_id = ? AND is_deleted = 0
                   ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END, version DESC""",
                (dataset_id, branch_id, head_version_id or ""),
            )
        else:
            rows = self.db.fetchall(
                """SELECT * FROM dataset_versions
                   WHERE dataset_id = ? AND is_deleted = 0
                   ORDER BY version DESC""",
                (dataset_id,),
            )
        return [self._row_to_version(r) for r in rows]

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def _row_to_branch(self, row: Any) -> DatasetBranch:
        d = dict(row)
        d["is_default"] = bool(d.get("is_default", 0))
        d["is_deleted"] = bool(d.get("is_deleted", 0))
        return DatasetBranch(**d)

    def _get_default_branch_id(self, dataset_id: str) -> str | None:
        row = self.db.fetchone(
            "SELECT default_branch_id FROM datasets WHERE id = ?", (dataset_id,)
        )
        return row["default_branch_id"] if row else None

    def _resolve_head_version(self, branch_id: str) -> DatasetVersion | None:
        row = self.db.fetchone(
            """SELECT v.* FROM dataset_branches b
               JOIN dataset_versions v ON v.id = b.head_version_id
               WHERE b.id = ? AND v.is_deleted = 0""",
            (branch_id,),
        )
        return self._row_to_version(row) if row else None

    def _set_branch_head_version(self, branch_id: str, version_id: str | None) -> None:
        self.db.execute(
            "UPDATE dataset_branches SET head_version_id = ? WHERE id = ?",
            (version_id, branch_id),
        )

    def _get_or_create_main_branch(self, dataset_id: str, author: str = "") -> DatasetBranch:
        """Return the default branch, creating it if missing (migration helper)."""
        default_id = self._get_default_branch_id(dataset_id)
        if default_id:
            branch = self.get_branch(default_id)
            if branch and not branch.is_deleted:
                return branch

        # Look for any existing "main" branch
        row = self.db.fetchone(
            "SELECT * FROM dataset_branches WHERE dataset_id = ? AND name = 'main' AND is_deleted = 0",
            (dataset_id,),
        )
        if row:
            branch = self._row_to_branch(row)
            # Set as default
            self.db.execute(
                "UPDATE dataset_branches SET is_default = 1 WHERE id = ?", (branch.id,)
            )
            self.db.execute(
                "UPDATE datasets SET default_branch_id = ? WHERE id = ?",
                (branch.id, dataset_id),
            )
            if not branch.head_version_id:
                head_row = self.db.fetchone(
                    """SELECT id FROM dataset_versions
                       WHERE dataset_id = ? AND branch_id = ? AND is_deleted = 0
                       ORDER BY version DESC LIMIT 1""",
                    (dataset_id, branch.id),
                )
                if head_row:
                    self._set_branch_head_version(branch.id, head_row["id"])
                    branch = DatasetBranch(**{**branch.model_dump(), "head_version_id": head_row["id"]})
            self.db.commit()
            return DatasetBranch(**{**branch.model_dump(), "is_default": True})

        # Create main branch from scratch
        now = _utcnow()
        branch_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO dataset_branches
               (id, dataset_id, name, description, base_version_id, head_version_id, author, created_at, is_default, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (branch_id, dataset_id, "main", "", None, None, author, now, 1, 0),
        )
        # Assign all unassigned versions to this branch
        self.db.execute(
            "UPDATE dataset_versions SET branch_id = ? WHERE dataset_id = ? AND branch_id IS NULL",
            (branch_id, dataset_id),
        )
        head_row = self.db.fetchone(
            """SELECT id FROM dataset_versions
               WHERE dataset_id = ? AND branch_id = ? AND is_deleted = 0
               ORDER BY version DESC LIMIT 1""",
            (dataset_id, branch_id),
        )
        head_version_id = head_row["id"] if head_row else None
        if head_version_id:
            self._set_branch_head_version(branch_id, head_version_id)
        self.db.execute(
            "UPDATE datasets SET default_branch_id = ? WHERE id = ?",
            (branch_id, dataset_id),
        )
        self.db.commit()
        return DatasetBranch(
            id=branch_id, dataset_id=dataset_id, name="main", description="",
            base_version_id=None, author=author, created_at=now,
            head_version_id=head_version_id, is_default=True, is_deleted=False,
        )

    def list_branches(self, dataset_id: str) -> list[DatasetBranch]:
        """List all active branches for a dataset."""
        rows = self.db.fetchall(
            "SELECT * FROM dataset_branches WHERE dataset_id = ? AND is_deleted = 0 ORDER BY created_at",
            (dataset_id,),
        )
        return [self._row_to_branch(r) for r in rows]

    def get_branch(self, branch_id: str) -> DatasetBranch | None:
        """Get a branch by ID."""
        row = self.db.fetchone(
            "SELECT * FROM dataset_branches WHERE id = ?", (branch_id,)
        )
        return self._row_to_branch(row) if row else None

    def create_branch(
        self,
        dataset_id: str,
        name: str,
        from_version_id: str | None = None,
        author: str = "",
        description: str = "",
    ) -> DatasetBranch:
        """Create a new branch from a specific version (or from the default branch head).

        Creates a forked version on the new branch that points to the same data file
        as the base version. This gives each branch its own version history.
        """
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        if not name or not name.strip():
            raise ValueError("Branch name cannot be empty")
        name = name.strip()

        # Check name uniqueness (including soft-deleted branches due to UNIQUE constraint)
        existing = self.db.fetchone(
            "SELECT id, is_deleted FROM dataset_branches WHERE dataset_id = ? AND name = ?",
            (dataset_id, name),
        )
        if existing:
            if existing["is_deleted"]:
                # Branch was soft-deleted - rename it to free up the name
                timestamp = _utcnow().replace(":", "-").replace("+", "at")
                deleted_rename = f"{name}-deleted-{timestamp}"
                self.db.execute(
                    "UPDATE dataset_branches SET name = ? WHERE id = ?",
                    (deleted_rename, existing["id"]),
                )
                log.info("renamed_deleted_branch", old_name=name, new_name=deleted_rename)
            else:
                raise ValueError(f"Branch '{name}' already exists")

        # Validate from_version_id and get the version to fork from
        base_version_id = from_version_id
        if from_version_id:
            vrow = self.db.fetchone(
                "SELECT * FROM dataset_versions WHERE id = ? AND dataset_id = ? AND is_deleted = 0",
                (from_version_id, dataset_id),
            )
            if not vrow:
                raise ValueError(f"Version not found: {from_version_id}")
        else:
            # No from_version_id specified: use the default branch's head
            default_branch_id = self._get_default_branch_id(dataset_id)
            if default_branch_id:
                head = self.get_branch_head_version(default_branch_id)
                if head:
                    base_version_id = head.id

        # Get the base version data for forking
        base_row = None
        if base_version_id:
            base_row = self.db.fetchone(
                "SELECT * FROM dataset_versions WHERE id = ?",
                (base_version_id,),
            )
            if not base_row:
                raise ValueError(f"Base version not found: {base_version_id}")

        now = _utcnow()
        branch_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO dataset_branches
               (id, dataset_id, name, description, base_version_id, head_version_id, author, created_at, is_default, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (branch_id, dataset_id, name, description, base_version_id, None, author, now, 0, 0),
        )

        fork_version_id: str | None = None
        # Create a forked version on the new branch
        if base_row:
            # Get next global version number
            max_ver_row = self.db.fetchone(
                "SELECT MAX(version) as max_ver FROM dataset_versions WHERE dataset_id = ?",
                (dataset_id,),
            )
            new_version = (max_ver_row["max_ver"] or 0) + 1

            fork_version_id = str(uuid.uuid4())
            # Get column_roles from base_row (may be None for old versions)
            column_roles = base_row["column_roles"] if base_row["column_roles"] else "{}"

            # Get the parent branch name for the fork reason
            parent_branch_name = "unknown"
            if base_row["branch_id"]:
                parent_branch = self.get_branch(base_row["branch_id"])
                if parent_branch:
                    parent_branch_name = parent_branch.name

            self.db.execute(
                """INSERT INTO dataset_versions
                (id, dataset_id, version, created_at, author, reason, sha256,
                 row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted, is_fork)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fork_version_id, dataset_id, new_version, base_row["created_at"], author,
                 f"Forked from {parent_branch_name}", base_row["sha256"], base_row["row_count"],
                 base_row["file_size_bytes"], base_row["storage_path"], branch_id,
                 column_roles, 0, 1),  # is_fork = 1
            )
            self.db.execute(
                "UPDATE dataset_branches SET head_version_id = ? WHERE id = ?",
                (fork_version_id, branch_id),
            )
            log.info("branch_version_forked", branch_id=branch_id, fork_version_id=fork_version_id,
                     from_version=base_version_id)

        self.db.commit()
        log.info("branch_created", dataset_id=dataset_id, branch=name, from_version=base_version_id)
        return DatasetBranch(
            id=branch_id, dataset_id=dataset_id, name=name, description=description,
            base_version_id=base_version_id, author=author, created_at=now,
            head_version_id=fork_version_id, is_default=False, is_deleted=False,
        )

    def update_branch(
        self,
        dataset_id: str,
        branch_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> DatasetBranch:
        """Update branch name and/or description."""
        branch = self.get_branch(branch_id)
        if branch is None or branch.dataset_id != dataset_id or branch.is_deleted:
            raise ValueError(f"Branch not found: {branch_id}")

        updates: list[str] = []
        params: list[Any] = []

        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Branch name cannot be empty")
            if name != branch.name:
                existing = self.db.fetchone(
                    "SELECT id, is_deleted FROM dataset_branches WHERE dataset_id = ? AND name = ? AND id != ?",
                    (dataset_id, name, branch_id),
                )
                if existing:
                    if existing["is_deleted"]:
                        # Branch was soft-deleted - rename it to free up the name
                        timestamp = _utcnow().replace(":", "-").replace("+", "at")
                        deleted_rename = f"{name}-deleted-{timestamp}"
                        self.db.execute(
                            "UPDATE dataset_branches SET name = ? WHERE id = ?",
                            (deleted_rename, existing["id"]),
                        )
                        log.info("renamed_deleted_branch", old_name=name, new_name=deleted_rename)
                    else:
                        raise ValueError(f"Branch name '{name}' already exists")
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if updates:
            params.append(branch_id)
            self.db.execute(
                f"UPDATE dataset_branches SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            self.db.commit()

        updated = self.get_branch(branch_id)
        assert updated is not None
        return updated

    def delete_branch(self, dataset_id: str, branch_id: str) -> dict[str, Any]:
        """Soft-delete a non-default branch and all its versions."""
        branch = self.get_branch(branch_id)
        if branch is None or branch.dataset_id != dataset_id or branch.is_deleted:
            raise ValueError(f"Branch not found: {branch_id}")
        if branch.is_default:
            raise ValueError("Cannot delete the default branch")

        # Soft-delete all versions belonging to this branch (including forked versions)
        self.db.execute(
            "UPDATE dataset_versions SET is_deleted = 1 WHERE branch_id = ?",
            (branch_id,),
        )
        self.db.execute(
            "UPDATE dataset_branches SET is_deleted = 1 WHERE id = ?", (branch_id,)
        )
        self.db.commit()
        log.info("branch_deleted", branch_id=branch_id)
        return {"deleted": True, "branch_id": branch_id}

    def set_default_branch(self, dataset_id: str, branch_id: str) -> dict[str, Any]:
        """Set a branch as the default and update current_version pointer."""
        branch = self.get_branch(branch_id)
        if branch is None or branch.dataset_id != dataset_id or branch.is_deleted:
            raise ValueError(f"Branch not found: {branch_id}")

        # Unset all defaults for this dataset
        self.db.execute(
            "UPDATE dataset_branches SET is_default = 0 WHERE dataset_id = ?",
            (dataset_id,),
        )
        # Set new default
        self.db.execute(
            "UPDATE dataset_branches SET is_default = 1 WHERE id = ?", (branch_id,)
        )
        self.db.execute(
            "UPDATE datasets SET default_branch_id = ? WHERE id = ?",
            (branch_id, dataset_id),
        )

        # Update current_version to head of the new default branch
        head = self.get_branch_head_version(branch_id)
        if head:
            csv_path = Path(head.storage_path)
            try:
                df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
                schema_info = _infer_schema(df)
                schema_json = orjson.dumps([s.model_dump() for s in schema_info]).decode()
            except Exception:
                schema_json = "{}"
            self.db.execute(
                """UPDATE datasets SET current_version = ?, row_count = ?,
                   file_size_bytes = ?, sha256 = ?, schema_info = ? WHERE id = ?""",
                (head.version, head.row_count, head.file_size_bytes,
                 head.sha256, schema_json, dataset_id),
            )
        self.db.commit()
        log.info("default_branch_set", dataset_id=dataset_id, branch_id=branch_id)
        return {"dataset_id": dataset_id, "default_branch_id": branch_id}

    def move_version_to_branch(
        self,
        dataset_id: str,
        version_id: str,
        target_branch_id: str,
        author: str = "",
    ) -> DatasetVersion:
        """Move a version to a different branch.

        This changes the branch association of an existing version. The version's
        data file is not modified - only the branch_id reference is updated.
        """
        # Validate the version exists
        version = self.get_version_by_id(version_id)
        if version is None:
            raise ValueError(f"Version not found: {version_id}")
        if version.dataset_id != dataset_id:
            raise ValueError(f"Version does not belong to dataset: {dataset_id}")

        # Validate target branch exists
        target_branch = self.get_branch(target_branch_id)
        if target_branch is None or target_branch.dataset_id != dataset_id or target_branch.is_deleted:
            raise ValueError(f"Target branch not found: {target_branch_id}")

        # Don't allow moving if already on target branch
        if version.branch_id == target_branch_id:
            return version

        # Prevent creating an empty branch via move; each active branch must keep at least one version.
        count_row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM dataset_versions WHERE branch_id = ? AND is_deleted = 0",
            (version.branch_id,),
        )
        cnt = count_row["cnt"] if count_row else 0
        if cnt <= 1:
            raise ValueError(
                "Cannot move the only version on a branch. "
                "Create a new version first, or delete the branch."
            )

        # Update the version's branch
        self.db.execute(
            "UPDATE dataset_versions SET branch_id = ? WHERE id = ?",
            (target_branch_id, version_id),
        )

        # If we moved the source head away, re-point source head to its latest remaining version.
        source_branch = self.get_branch(version.branch_id) if version.branch_id else None
        if source_branch and source_branch.head_version_id == version_id:
            new_source_head_row = self.db.fetchone(
                """SELECT id FROM dataset_versions
                   WHERE branch_id = ? AND is_deleted = 0
                   ORDER BY version DESC LIMIT 1""",
                (version.branch_id,),
            )
            self._set_branch_head_version(
                version.branch_id,
                new_source_head_row["id"] if new_source_head_row else None,
            )

        # For target branch, preserve manual head unless this move is newer or target has no head.
        target_head = self.get_branch_head_version(target_branch_id)
        if target_head is None or version.version >= target_head.version:
            self._set_branch_head_version(target_branch_id, version_id)

        # Keep dataset current_version in sync with the default branch head after branch moves.
        default_branch_id = self._get_default_branch_id(dataset_id)
        if default_branch_id and (
            version.branch_id == default_branch_id or target_branch_id == default_branch_id
        ):
            default_head = self.get_branch_head_version(default_branch_id)
            if default_head:
                self.db.execute(
                    """UPDATE datasets SET current_version = ?, row_count = ?,
                       file_size_bytes = ?, sha256 = ? WHERE id = ?""",
                    (
                        default_head.version,
                        default_head.row_count,
                        default_head.file_size_bytes,
                        default_head.sha256,
                        dataset_id,
                    ),
                )
        self.db.commit()

        # Fetch and return the updated version
        updated = self.get_version_by_id(version_id)
        assert updated is not None
        log.info("version_moved_to_branch", version_id=version_id,
                 from_branch=version.branch_id, to_branch=target_branch_id, author=author)
        return updated

    def copy_version(
        self,
        dataset_id: str,
        version_id: str,
        new_reason: str,
        author: str = "",
        branch_id: str | None = None,
    ) -> DatasetVersion:
        """Create a copy of a version with modified metadata.

        The copied version shares the same data file (storage_path) but has:
        - New UUID and version number
        - New created_at timestamp
        - Custom reason (from request)
        - Optional custom author (from request)
        - Same sha256, row_count, file_size_bytes
        - Same column_roles
        - Marked as is_fork=True

        This is useful for creating labeled snapshots of the same data state.
        """
        # Validate source version exists
        source_version = self.get_version_by_id(version_id)
        if source_version is None:
            raise ValueError(f"Version not found: {version_id}")
        if source_version.dataset_id != dataset_id:
            raise ValueError(f"Version does not belong to dataset: {dataset_id}")

        # Resolve target branch
        if branch_id is None:
            branch = self._get_or_create_main_branch(dataset_id, author)
            branch_id = branch.id
        else:
            branch = self.get_branch(branch_id)
            if branch is None or branch.is_deleted:
                raise ValueError(f"Branch not found: {branch_id}")

        # Get next global version number
        max_ver_row = self.db.fetchone(
            "SELECT MAX(version) as max_ver FROM dataset_versions WHERE dataset_id = ?",
            (dataset_id,),
        )
        new_ver = (max_ver_row["max_ver"] or 0) + 1

        # Create copy with new metadata
        now = _utcnow()
        copy_version_id = str(uuid.uuid4())

        self.db.execute(
            """INSERT INTO dataset_versions
            (id, dataset_id, version, created_at, author, reason, sha256,
             row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted, is_fork)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (copy_version_id, dataset_id, new_ver, now, author, new_reason,
             source_version.sha256, source_version.row_count,
             source_version.file_size_bytes, source_version.storage_path,
             branch_id, orjson.dumps(source_version.column_roles).decode(), 0, 1),
        )
        self.db.commit()

        log.info("version_copied", dataset_id=dataset_id, version=new_ver,
                 source_version_id=version_id, branch_id=branch_id, author=author)

        # Fetch and return the copied version
        copied = self.get_version_by_id(copy_version_id)
        assert copied is not None
        return copied

    def restore_version(
        self,
        dataset_id: str,
        version_id: str,
        reason: str = "",
        author: str = "",
    ) -> DatasetVersion:
        """Restore a historical version by creating a new head on the same branch.

        The restored version keeps the same underlying data (storage_path/hash),
        but is recorded as a fresh immutable version entry with a new version number.
        """
        src_row = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ? AND dataset_id = ? AND is_deleted = 0",
            (version_id, dataset_id),
        )
        if src_row is None:
            raise ValueError(f"Version not found: {version_id}")
        source = self._row_to_version(src_row)

        if source.branch_id is None:
            raise ValueError("Source version is not associated with a branch")
        branch = self.get_branch(source.branch_id)
        if branch is None or branch.dataset_id != dataset_id or branch.is_deleted:
            raise ValueError(f"Branch not found: {source.branch_id}")

        max_ver_row = self.db.fetchone(
            "SELECT MAX(version) as max_ver FROM dataset_versions WHERE dataset_id = ?",
            (dataset_id,),
        )
        new_ver = (max_ver_row["max_ver"] or 0) + 1
        now = _utcnow()
        restored_id = str(uuid.uuid4())
        restored_reason = reason.strip() or f"Restored from v{source.version}"

        self.db.execute(
            """INSERT INTO dataset_versions
            (id, dataset_id, version, created_at, author, reason, sha256,
             row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted, is_fork)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                restored_id,
                dataset_id,
                new_ver,
                now,
                author,
                restored_reason,
                source.sha256,
                source.row_count,
                source.file_size_bytes,
                source.storage_path,
                source.branch_id,
                orjson.dumps(source.column_roles).decode(),
                0,
                0,
            ),
        )

        # If restoring on default branch, dataset's current pointer should follow.
        default_branch_id = self._get_default_branch_id(dataset_id)
        if source.branch_id == default_branch_id:
            self.db.execute(
                """UPDATE datasets SET current_version = ?, row_count = ?,
                   file_size_bytes = ?, sha256 = ? WHERE id = ?""",
                (new_ver, source.row_count, source.file_size_bytes, source.sha256, dataset_id),
            )

        self.db.commit()
        restored = self.get_version_by_id(restored_id)
        assert restored is not None
        log.info(
            "version_restored",
            dataset_id=dataset_id,
            source_version_id=version_id,
            restored_version_id=restored_id,
            branch_id=source.branch_id,
            author=author,
        )
        return restored

    def set_version_as_default(
        self,
        dataset_id: str,
        version_id: str,
        author: str = "",
    ) -> DatasetVersion:
        """Make an existing version the default dataset version (no new version created)."""
        src_row = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ? AND dataset_id = ? AND is_deleted = 0",
            (version_id, dataset_id),
        )
        if src_row is None:
            raise ValueError(f"Version not found: {version_id}")
        source = self._row_to_version(src_row)

        if source.branch_id is None:
            raise ValueError("Source version is not associated with a branch")
        branch = self.get_branch(source.branch_id)
        if branch is None or branch.dataset_id != dataset_id or branch.is_deleted:
            raise ValueError(f"Branch not found: {source.branch_id}")

        # Switch dataset default branch without touching branch head pointer.
        self.db.execute(
            "UPDATE dataset_branches SET is_default = 0 WHERE dataset_id = ?",
            (dataset_id,),
        )
        self.db.execute(
            "UPDATE dataset_branches SET is_default = 1 WHERE id = ?",
            (source.branch_id,),
        )
        self.db.execute(
            "UPDATE datasets SET default_branch_id = ? WHERE id = ?",
            (source.branch_id, dataset_id),
        )

        csv_path = Path(source.storage_path)
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
            schema_info = _infer_schema(df)
            schema_json = orjson.dumps([s.model_dump() for s in schema_info]).decode()
        except Exception:
            schema_json = "{}"
        self.db.execute(
            """UPDATE datasets SET current_version = ?, row_count = ?,
               file_size_bytes = ?, sha256 = ?, schema_info = ? WHERE id = ?""",
            (source.version, source.row_count, source.file_size_bytes, source.sha256, schema_json, dataset_id),
        )
        self.db.commit()
        log.info("version_set_as_default", dataset_id=dataset_id, version_id=version_id, branch_id=source.branch_id, author=author)
        return source

    def get_branch_head_version(self, branch_id: str) -> DatasetVersion | None:
        """Get the selected head version on a branch, with fallback to latest by version."""
        explicit = self._resolve_head_version(branch_id)
        if explicit is not None:
            return explicit

        row = self.db.fetchone(
            """SELECT * FROM dataset_versions
               WHERE branch_id = ? AND is_deleted = 0
               ORDER BY version DESC LIMIT 1""",
            (branch_id,),
        )
        return self._row_to_version(row) if row else None

    # ------------------------------------------------------------------
    # Version metadata editing / deletion
    # ------------------------------------------------------------------

    def get_version_by_id(self, version_id: str) -> DatasetVersion | None:
        """Get a version by UUID."""
        row = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)
        )
        return self._row_to_version(row) if row else None

    def update_version_metadata(
        self,
        dataset_id: str,
        version_id: str,
        reason: str,
    ) -> DatasetVersion:
        """Update the 'reason' field of a version record."""
        row = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ? AND dataset_id = ?",
            (version_id, dataset_id),
        )
        if row is None:
            raise ValueError(f"Version not found: {version_id}")

        self.db.execute(
            "UPDATE dataset_versions SET reason = ? WHERE id = ?",
            (reason, version_id),
        )
        self.db.commit()
        updated = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)
        )
        assert updated is not None
        return self._row_to_version(updated)

    def delete_version(self, dataset_id: str, version_id: str) -> dict[str, Any]:
        """Soft-delete a version. Cannot delete if it is the only version on its branch."""
        row = self.db.fetchone(
            "SELECT * FROM dataset_versions WHERE id = ? AND dataset_id = ? AND is_deleted = 0",
            (version_id, dataset_id),
        )
        if row is None:
            raise ValueError(f"Version not found: {version_id}")

        ver = self._row_to_version(row)

        # Must have at least one other non-deleted version on the same branch
        count_row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM dataset_versions WHERE branch_id = ? AND is_deleted = 0",
            (ver.branch_id,),
        )
        cnt = count_row["cnt"] if count_row else 0
        if cnt <= 1:
            raise ValueError(
                "Cannot delete the only version on a branch. "
                "Delete the branch instead, or create a new version first."
            )

        # Soft-delete
        self.db.execute(
            "UPDATE dataset_versions SET is_deleted = 1 WHERE id = ?", (version_id,)
        )

        # If this was the branch head, promote the next latest version on that branch.
        branch = self.get_branch(ver.branch_id) if ver.branch_id else None
        if branch and branch.head_version_id == version_id:
            new_head_row = self.db.fetchone(
                """SELECT id FROM dataset_versions
                   WHERE branch_id = ? AND is_deleted = 0
                   ORDER BY version DESC LIMIT 1""",
                (ver.branch_id,),
            )
            self._set_branch_head_version(
                ver.branch_id,
                new_head_row["id"] if new_head_row else None,
            )

        # If this was head of the default branch, update current_version
        default_branch_id = self._get_default_branch_id(dataset_id)
        if ver.branch_id == default_branch_id:
            new_head_row = self.db.fetchone(
                """SELECT * FROM dataset_versions
                   WHERE branch_id = ? AND is_deleted = 0
                   ORDER BY version DESC LIMIT 1""",
                (ver.branch_id,),
            )
            if new_head_row:
                new_head = self._row_to_version(new_head_row)
                self.db.execute(
                    """UPDATE datasets SET current_version = ?, row_count = ?,
                       file_size_bytes = ?, sha256 = ? WHERE id = ?""",
                    (new_head.version, new_head.row_count, new_head.file_size_bytes,
                     new_head.sha256, dataset_id),
                )

        self.db.commit()
        log.info("version_deleted", version_id=version_id, dataset_id=dataset_id)
        return {"deleted": True, "version_id": version_id}

    def get_column_roles(
        self,
        dataset_id: str,
        version_id: str | None = None,
        version: int | None = None,
    ) -> dict[str, str]:
        """Get column roles for a version.

        Lookup priority:
        1. version_id (UUID) if provided.
        2. version (integer) if provided.
        3. current_version of the dataset (default).
        """
        if version_id:
            row = self.db.fetchone(
                "SELECT column_roles FROM dataset_versions WHERE id = ? AND dataset_id = ?",
                (version_id, dataset_id),
            )
        elif version is not None:
            row = self.db.fetchone(
                """SELECT column_roles FROM dataset_versions
                   WHERE dataset_id = ? AND version = ? AND is_deleted = 0""",
                (dataset_id, version),
            )
        else:
            ds = self.get_dataset(dataset_id)
            if ds is None:
                raise ValueError(f"Dataset not found: {dataset_id}")
            row = self.db.fetchone(
                """SELECT column_roles FROM dataset_versions
                   WHERE dataset_id = ? AND version = ? AND is_deleted = 0""",
                (dataset_id, ds.current_version),
            )
        if row is None:
            return {}
        cr = row["column_roles"]
        if isinstance(cr, str):
            try:
                return orjson.loads(cr)
            except Exception:
                return {}
        return cr or {}

    # ------------------------------------------------------------------
    # Update dataset metadata
    # ------------------------------------------------------------------

    def update_metadata(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> DatasetMeta | None:
        """Update dataset metadata fields."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            return None

        updates: list[str] = []
        params: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if tags is not None:
            updates.append("tags = ?")
            params.append(orjson.dumps(tags).decode())

        if not updates:
            return ds

        params.append(dataset_id)
        self.db.execute(
            f"UPDATE datasets SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        self.db.commit()
        return self.get_dataset(dataset_id)

    # ------------------------------------------------------------------
    # Delete dataset
    # ------------------------------------------------------------------

    def get_dependencies(self, dataset_id: str) -> dict[str, int]:
        """Check how many models and analyses depend on this dataset."""
        models_row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM models WHERE dataset_id = ? AND status = 'active'",
            (dataset_id,),
        )
        analyses_row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM analysis_runs WHERE dataset_id = ?",
            (dataset_id,),
        )
        return {
            "models": models_row["cnt"] if models_row else 0,
            "analyses": analyses_row["cnt"] if analyses_row else 0,
        }

    def delete_dataset(self, dataset_id: str, force: bool = False) -> dict[str, Any]:
        """Soft-delete a dataset. Returns dependency info. Use force=True to delete despite deps."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        deps = self.get_dependencies(dataset_id)
        if not force and (deps["models"] > 0 or deps["analyses"] > 0):
            return {
                "deleted": False,
                "reason": "Dataset has dependencies",
                "dependencies": deps,
            }

        self.db.execute(
            "UPDATE datasets SET status = 'deleted' WHERE id = ?",
            (dataset_id,),
        )
        self.db.commit()
        log.info("dataset_deleted", id=dataset_id, force=force)
        return {"deleted": True, "dependencies": deps}

    # ------------------------------------------------------------------
    # Subset
    # ------------------------------------------------------------------

    def create_subset(
        self,
        dataset_id: str,
        filter_config: dict[str, Any],
        name: str,
        description: str = "",
        author: str = "",
        version: int | None = None,
    ) -> DatasetMeta:
        """Create a new dataset from a filtered subset of an existing one."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        v = version or ds.current_version
        file_path = self._get_version_file(dataset_id, v)
        df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        df = self._apply_filters(df, filter_config)

        if df.empty:
            raise ValueError("Filter resulted in an empty dataset")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)

        try:
            subset_desc = f"Subset of '{ds.name}' (v{v}). {description}".strip()
            return self.upload_dataset(
                file_path=tmp_path,
                name=name,
                description=subset_desc,
                tags=ds.tags + ["subset"],
                author=author,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Version editing
    # ------------------------------------------------------------------

    def create_version(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        reason: str = "manual edit",
        author: str = "",
        branch_id: str | None = None,
        column_roles: dict[str, str] | None = None,
    ) -> DatasetVersion:
        """Save df as a new immutable version of dataset_id on the specified branch."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Resolve branch
        if branch_id is None:
            branch = self._get_or_create_main_branch(dataset_id, author)
            branch_id = branch.id
        else:
            branch = self.get_branch(branch_id)
            if branch is None or branch.is_deleted:
                raise ValueError(f"Branch not found: {branch_id}")

        # Get next global version number (max across all branches)
        max_ver_row = self.db.fetchone(
            "SELECT MAX(version) as max_ver FROM dataset_versions WHERE dataset_id = ?",
            (dataset_id,),
        )
        new_ver = (max_ver_row["max_ver"] or 0) + 1

        version_dir = self.datasets_dir / dataset_id / f"v{new_ver}"
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / "data.csv"
        df.to_csv(dest, index=False)

        sha = _sha256(dest)
        file_size = dest.stat().st_size
        schema_info = _infer_schema(df)
        now = _utcnow()
        version_id = str(uuid.uuid4())

        # Inherit or detect column_roles
        if column_roles is None:
            head = self.get_branch_head_version(branch_id)
            if head is not None:
                column_roles = dict(head.column_roles)
            else:
                column_roles = _detect_initial_column_roles(df)

        version_meta = {
            "id": version_id,
            "dataset_id": dataset_id,
            "version": new_ver,
            "created_at": now,
            "author": author,
            "reason": reason,
            "sha256": sha,
            "row_count": len(df),
            "file_size_bytes": file_size,
        }
        (version_dir / "metadata.json").write_bytes(
            orjson.dumps(version_meta, option=orjson.OPT_INDENT_2)
        )

        schema_json = orjson.dumps([s.model_dump() for s in schema_info]).decode()
        column_roles_json = orjson.dumps(column_roles).decode()

        self.db.execute(
            """INSERT INTO dataset_versions
            (id, dataset_id, version, created_at, author, reason, sha256,
             row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (version_id, dataset_id, new_ver, now, author, reason,
             sha, len(df), file_size, str(dest), branch_id, column_roles_json, 0),
        )
        self._set_branch_head_version(branch_id, version_id)

        # Update datasets.current_version only if this is on the default branch
        default_branch_id = self._get_default_branch_id(dataset_id)
        if branch_id == default_branch_id:
            self.db.execute(
                """UPDATE datasets SET current_version = ?, row_count = ?,
                   file_size_bytes = ?, sha256 = ?, schema_info = ?
                   WHERE id = ?""",
                (new_ver, len(df), file_size, sha, schema_json, dataset_id),
            )

        self.db.commit()

        log.info("dataset_version_created", dataset_id=dataset_id, version=new_ver,
                 branch_id=branch_id, reason=reason)
        return DatasetVersion(
            id=version_id,
            dataset_id=dataset_id,
            version=new_ver,
            created_at=now,
            author=author,
            reason=reason,
            sha256=sha,
            row_count=len(df),
            file_size_bytes=file_size,
            storage_path=str(dest),
            branch_id=branch_id,
            column_roles=column_roles,
        )

    def _load_branch_head_df(
        self, dataset_id: str, branch_id: str | None, author: str
    ) -> tuple[pd.DataFrame, str]:
        """Load DataFrame from the head of the specified (or default) branch.
        Returns (df, resolved_branch_id).
        """
        if branch_id is None:
            branch = self._get_or_create_main_branch(dataset_id, author)
            branch_id = branch.id

        head = self.get_branch_head_version(branch_id)
        if head is None:
            # Fallback for old branches without forked versions (backward compatibility)
            br = self.get_branch(branch_id)
            if br and br.base_version_id:
                brow = self.db.fetchone(
                    "SELECT storage_path FROM dataset_versions WHERE id = ?",
                    (br.base_version_id,),
                )
                if brow:
                    file_path = Path(brow["storage_path"])
                    df = pd.read_csv(file_path, encoding="utf-8", dtype=str, keep_default_na=False)
                    return df, branch_id
            # No data at all — this shouldn't happen for a valid branch
            ds = self.get_dataset(dataset_id)
            if ds is None:
                raise ValueError(f"Dataset not found: {dataset_id}")
            file_path = self._get_version_file(dataset_id, ds.current_version)
            df = pd.read_csv(file_path, encoding="utf-8", dtype=str, keep_default_na=False)
            return df, branch_id

        file_path = Path(head.storage_path)
        df = pd.read_csv(file_path, encoding="utf-8", dtype=str, keep_default_na=False)
        return df, branch_id

    def update_cells(
        self,
        dataset_id: str,
        changes: list[dict],
        reason: str = "cell edits",
        author: str = "",
        branch_id: str | None = None,
    ) -> DatasetVersion:
        """Apply cell-level edits and save as a new version on the given branch."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        df, branch_id = self._load_branch_head_df(dataset_id, branch_id, author)

        errors: list[str] = []
        for ch in changes:
            row_idx = ch["row_idx"]
            col = ch["col"]
            value = ch["value"]
            if col not in df.columns:
                errors.append(f"Column not found: {col!r}")
                continue
            if row_idx < 0 or row_idx >= len(df):
                errors.append(f"Row index out of range: {row_idx}")
                continue
            df.at[row_idx, col] = value

        if errors:
            raise ValueError(f"Cell update errors: {'; '.join(errors)}")

        return self.create_version(dataset_id, df, reason=reason, author=author,
                                   branch_id=branch_id)

    def add_rows(
        self,
        dataset_id: str,
        new_rows: list[dict],
        reason: str = "added rows",
        author: str = "",
        branch_id: str | None = None,
    ) -> DatasetVersion:
        """Append rows and save as a new version."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        df, branch_id = self._load_branch_head_df(dataset_id, branch_id, author)

        new_df = pd.DataFrame(new_rows)
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = ""
        new_df = new_df[df.columns].fillna("").astype(str)

        df = pd.concat([df, new_df], ignore_index=True)
        return self.create_version(dataset_id, df, reason=reason, author=author,
                                   branch_id=branch_id)

    def delete_rows(
        self,
        dataset_id: str,
        row_indices: list[int],
        reason: str = "deleted rows",
        author: str = "",
        branch_id: str | None = None,
    ) -> DatasetVersion:
        """Delete rows by absolute index and save as a new version."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        df, branch_id = self._load_branch_head_df(dataset_id, branch_id, author)

        invalid = [i for i in row_indices if i < 0 or i >= len(df)]
        if invalid:
            raise ValueError(f"Row indices out of range: {invalid}")

        df = df.drop(index=row_indices).reset_index(drop=True)

        if df.empty:
            raise ValueError("Cannot delete all rows — dataset would be empty")

        return self.create_version(dataset_id, df, reason=reason, author=author,
                                   branch_id=branch_id)

    def rename_columns(
        self,
        dataset_id: str,
        renames: dict[str, str],
        reason: str = "rename columns",
        author: str = "",
        branch_id: str | None = None,
    ) -> DatasetVersion:
        """Rename one or more columns and save as a new immutable version.
        Column roles are propagated automatically to reflect the new names.
        """
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        df, branch_id = self._load_branch_head_df(dataset_id, branch_id, author)

        unknown = [old for old in renames if old not in df.columns]
        if unknown:
            raise ValueError(f"Unknown columns: {', '.join(unknown)}")

        new_cols = [renames.get(c, c) for c in df.columns]
        if len(new_cols) != len(set(new_cols)):
            raise ValueError("Column rename would produce duplicate column names")

        # Propagate column_roles through the rename
        head = self.get_branch_head_version(branch_id) if branch_id else None
        existing_roles: dict[str, str] = {}
        if head is not None:
            existing_roles = dict(head.column_roles)
        elif ds.current_version:
            existing_roles = self.get_column_roles(dataset_id)
        new_roles = _propagate_column_roles(existing_roles, renames)

        df = df.rename(columns=renames)
        return self.create_version(dataset_id, df, reason=reason, author=author,
                                   branch_id=branch_id, column_roles=new_roles)

    def create_empty_dataset(
        self,
        name: str,
        columns: list[str],
        description: str = "",
        tags: list[str] | None = None,
        author: str = "",
    ) -> DatasetMeta:
        """Create a new dataset with given column names and zero rows."""
        if len(columns) < 2:
            raise ValueError("At least 2 columns required")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names")

        tags = tags or []
        dataset_id = str(uuid.uuid4())
        now = _utcnow()
        version = 1

        branch_id = str(uuid.uuid4())

        version_dir = self.datasets_dir / dataset_id / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / "data.csv"

        df = pd.DataFrame(columns=columns)
        df.to_csv(dest, index=False)

        sha = _sha256(dest)
        file_size = dest.stat().st_size
        schema_info = [
            ColumnSchema(name=col, dtype="object", n_unique=0, n_null=0, sample_values=[])
            for col in columns
        ]
        column_roles = _detect_initial_column_roles(df)

        version_id = str(uuid.uuid4())
        version_meta = {
            "id": version_id,
            "dataset_id": dataset_id,
            "version": version,
            "created_at": now,
            "author": author,
            "reason": "created empty dataset",
            "sha256": sha,
            "row_count": 0,
            "file_size_bytes": file_size,
        }
        (version_dir / "metadata.json").write_bytes(
            orjson.dumps(version_meta, option=orjson.OPT_INDENT_2)
        )

        schema_json = orjson.dumps([s.model_dump() for s in schema_info]).decode()
        tags_json = orjson.dumps(tags).decode()
        column_roles_json = orjson.dumps(column_roles).decode()

        self.db.execute(
            """INSERT INTO datasets
            (id, name, description, tags, author, created_at, current_version,
             schema_info, row_count, file_size_bytes, sha256, status, default_branch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, name, description, tags_json, author, now, version,
             schema_json, 0, file_size, sha, "active", branch_id),
        )
        self.db.execute(
            """INSERT INTO dataset_branches
               (id, dataset_id, name, description, base_version_id, head_version_id, author, created_at, is_default, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (branch_id, dataset_id, "main", "", None, version_id, author, now, 1, 0),
        )
        self.db.execute(
            """INSERT INTO dataset_versions
            (id, dataset_id, version, created_at, author, reason, sha256,
             row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (version_id, dataset_id, version, now, author, "created empty dataset",
             sha, 0, file_size, str(dest), branch_id, column_roles_json, 0),
        )
        self.db.commit()

        log.info("dataset_created_empty", id=dataset_id, name=name, columns=columns)
        return DatasetMeta(
            id=dataset_id,
            name=name,
            description=description,
            tags=tags,
            author=author,
            created_at=now,
            current_version=version,
            schema_info=schema_info,
            row_count=0,
            file_size_bytes=file_size,
            sha256=sha,
            default_branch_id=branch_id,
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_dataframe(
        self,
        dataset_id: str,
        version: int | None = None,
        branch_id: str | None = None,
    ) -> pd.DataFrame:
        """Load a dataset version as a DataFrame.

        If branch_id is given and version is None, loads the head of that branch.
        Applies COLUMN_MAP renaming (Russian → English), then normalises any
        user-renamed columns back to their standard pipeline names using the
        stored column_roles for that version.
        """
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Resolve file path and track which version we loaded (for column_roles).
        actual_version: int | None = None
        actual_version_id: str | None = None  # UUID; used only for fallback branches

        if version is not None:
            file_path = self._get_version_file(dataset_id, version)
            actual_version = version
        elif branch_id is not None:
            head = self.get_branch_head_version(branch_id)
            if head is None:
                # Fallback for old branches without forked versions (backward compatibility)
                branch = self.get_branch(branch_id)
                if branch and branch.base_version_id:
                    base_row = self.db.fetchone(
                        "SELECT * FROM dataset_versions WHERE id = ?",
                        (branch.base_version_id,),
                    )
                    if base_row:
                        file_path = Path(base_row["storage_path"])
                        actual_version_id = base_row["id"]
                    else:
                        raise ValueError(f"Base version not found: {branch.base_version_id}")
                else:
                    raise ValueError(f"No versions found on branch {branch_id}")
            else:
                file_path = self._get_version_file(dataset_id, head.version)
                actual_version = head.version
        else:
            file_path = self._get_version_file(dataset_id, ds.current_version)
            actual_version = ds.current_version

        df = pd.read_csv(file_path, encoding="utf-8", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=COLUMN_MAP)

        # Second pass: normalise user-renamed columns via column_roles.
        # e.g. if the user renamed "sentiment_class" → "Класс Тональности",
        # column_roles stores {"Класс Тональности": "sentiment"} and we rename
        # it back to the standard name "sentiment_class" expected by the pipeline.
        col_roles = self.get_column_roles(
            dataset_id,
            version_id=actual_version_id,
            version=actual_version,
        )
        role_renames = {
            col: _ROLE_TO_STANDARD[role]
            for col, role in col_roles.items()
            if role in _ROLE_TO_STANDARD
            and col in df.columns
            and col != _ROLE_TO_STANDARD[role]
        }
        if role_renames:
            df = df.rename(columns=role_renames)

        # Final safety net for legacy versions where column_roles were not persisted.
        fallback_renames = _fallback_standard_renames(df)
        if fallback_renames:
            df = df.rename(columns=fallback_renames)
            log.info(
                "dataset_column_aliases_normalized",
                renamed_count=len(fallback_renames),
                targets=sorted(set(fallback_renames.values())),
            )

        return df

    def get_csv_path(self, dataset_id: str, version: int | None = None) -> Path:
        """Get the file path of a dataset version CSV."""
        ds = self.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        v = version or ds.current_version
        return self._get_version_file(dataset_id, v)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version_file(self, dataset_id: str, version: int) -> Path:
        """Get the CSV file path for a specific dataset version."""
        row = self.db.fetchone(
            "SELECT storage_path FROM dataset_versions WHERE dataset_id = ? AND version = ?",
            (dataset_id, version),
        )
        if row:
            return Path(row["storage_path"])
        return self.datasets_dir / dataset_id / f"v{version}" / "data.csv"

    def _row_to_version(self, row: Any) -> DatasetVersion:
        """Parse a SQLite Row into a DatasetVersion, handling JSON fields."""
        d = dict(row)
        # Parse column_roles from JSON string
        cr = d.get("column_roles")
        if isinstance(cr, str):
            try:
                d["column_roles"] = orjson.loads(cr)
            except Exception:
                d["column_roles"] = {}
        elif cr is None:
            d["column_roles"] = {}
        # Convert is_fork from int to bool (SQLite stores INTEGER)
        if "is_fork" in d:
            d["is_fork"] = bool(d.get("is_fork", 0))
        else:
            d["is_fork"] = False  # Default for old rows without the column
        # Drop columns not in the model
        d.pop("schema_info", None)
        d.pop("is_deleted", None)
        return DatasetVersion(**d)

    def _apply_filters(self, df: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """Apply filter configuration to a DataFrame."""
        if "column_equals" in filter_config:
            for col, val in filter_config["column_equals"].items():
                if col in df.columns:
                    df = df[df[col].astype(str) == str(val)]

        if "column_in" in filter_config:
            for col, vals in filter_config["column_in"].items():
                if col in df.columns:
                    df = df[df[col].astype(str).isin([str(v) for v in vals])]

        if "column_contains" in filter_config:
            for col, substring in filter_config["column_contains"].items():
                if col in df.columns:
                    df = df[df[col].astype(str).str.contains(str(substring), case=False, na=False)]

        if "row_indices" in filter_config:
            indices = filter_config["row_indices"]
            valid = [i for i in indices if 0 <= i < len(df)]
            df = df.iloc[valid]

        return df

    @staticmethod
    def _row_to_dataset(row: Any) -> DatasetMeta:
        """Convert a SQLite Row to a DatasetMeta."""
        d = dict(row)
        d["tags"] = orjson.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
        d["schema_info"] = orjson.loads(d["schema_info"]) if isinstance(d["schema_info"], str) else d["schema_info"]
        if isinstance(d["schema_info"], list):
            d["schema_info"] = [
                ColumnSchema(**s) if isinstance(s, dict) else s
                for s in d["schema_info"]
            ]
        # default_branch_id may not exist in old rows
        d.setdefault("default_branch_id", None)
        return DatasetMeta(**d)
