"""SQLite connection manager and schema creation."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    tags              TEXT NOT NULL DEFAULT '[]',      -- JSON array
    author            TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    current_version   INTEGER NOT NULL DEFAULT 1,
    schema_info       TEXT NOT NULL DEFAULT '{}',      -- JSON
    row_count         INTEGER NOT NULL DEFAULT 0,
    file_size_bytes   INTEGER NOT NULL DEFAULT 0,
    sha256            TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'active',  -- active | deleted
    default_branch_id TEXT
);

CREATE TABLE IF NOT EXISTS dataset_branches (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id),
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    base_version_id TEXT,              -- version_id this branch forked from (NULL for main)
    author          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    is_default      INTEGER NOT NULL DEFAULT 0,
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(dataset_id, name)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id),
    version         INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    author          TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT 'initial upload',
    sha256          TEXT NOT NULL DEFAULT '',
    row_count       INTEGER NOT NULL DEFAULT 0,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    storage_path    TEXT NOT NULL,
    branch_id       TEXT REFERENCES dataset_branches(id),
    column_roles    TEXT NOT NULL DEFAULT '{}',  -- JSON {col_name: role}
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    is_fork         INTEGER NOT NULL DEFAULT 0,  -- 1 if this is a forked version
    UNIQUE(dataset_id, version)
);

CREATE TABLE IF NOT EXISTS models (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    task            TEXT NOT NULL,           -- language | sentiment | detail_level
    model_type      TEXT NOT NULL,           -- tfidf | char_ngram
    version         INTEGER NOT NULL DEFAULT 1,
    dataset_id      TEXT,
    dataset_version INTEGER,
    config          TEXT NOT NULL DEFAULT '{}',     -- JSON
    metrics         TEXT NOT NULL DEFAULT '{}',     -- JSON
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active', -- active | archived
    storage_path    TEXT NOT NULL DEFAULT '',
    run_id          TEXT
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '[]',      -- JSON array
    comments        TEXT NOT NULL DEFAULT '',
    dataset_id      TEXT,
    dataset_version INTEGER,
    model_ids       TEXT NOT NULL DEFAULT '[]',      -- JSON array
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending', -- pending | running | completed | failed
    run_id          TEXT NOT NULL DEFAULT '',         -- link to pipeline run dir
    result_summary  TEXT NOT NULL DEFAULT '{}'        -- JSON
);

CREATE TABLE IF NOT EXISTS saved_filters (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,          -- analysis_results | datasets | models
    filter_config   TEXT NOT NULL DEFAULT '{}', -- JSON
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset ON dataset_versions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_models_task ON models(task);
CREATE INDEX IF NOT EXISTS idx_models_dataset ON models(dataset_id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_dataset ON analysis_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_status ON analysis_runs(status);
"""


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class Database:
    """Thread-safe SQLite connection manager with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._get_connection()

    def _init_schema(self) -> None:
        """Create tables if they don't exist, then run migrations."""
        conn = self._get_connection()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        self._run_migrations()
        log.info("database_initialized", path=str(self.db_path))

    def _run_migrations(self) -> None:
        """Apply incremental schema migrations for existing databases."""
        conn = self._get_connection()

        # Ensure dataset_branches table exists (may not on pre-Phase7 DBs)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dataset_branches (
                id              TEXT PRIMARY KEY,
                dataset_id      TEXT NOT NULL REFERENCES datasets(id),
                name            TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                base_version_id TEXT,
                author          TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL,
                is_default      INTEGER NOT NULL DEFAULT 0,
                is_deleted      INTEGER NOT NULL DEFAULT 0,
                UNIQUE(dataset_id, name)
            );
        """)

        # Add columns to datasets if missing
        existing_ds = {row[1] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        if "default_branch_id" not in existing_ds:
            conn.execute("ALTER TABLE datasets ADD COLUMN default_branch_id TEXT")

        # Add columns to dataset_versions if missing
        existing_ver = {row[1] for row in conn.execute("PRAGMA table_info(dataset_versions)").fetchall()}
        if "branch_id" not in existing_ver:
            conn.execute("ALTER TABLE dataset_versions ADD COLUMN branch_id TEXT")
        if "column_roles" not in existing_ver:
            conn.execute("ALTER TABLE dataset_versions ADD COLUMN column_roles TEXT NOT NULL DEFAULT '{}'")
        if "is_deleted" not in existing_ver:
            conn.execute("ALTER TABLE dataset_versions ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
        if "is_fork" not in existing_ver:
            conn.execute("ALTER TABLE dataset_versions ADD COLUMN is_fork INTEGER NOT NULL DEFAULT 0")

        # Create indexes that depend on migrated columns (safe to run after columns exist)
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_dataset_branches_dataset ON dataset_branches(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_dataset_versions_branch ON dataset_versions(branch_id);
        """)

        conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def commit(self) -> None:
        self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
