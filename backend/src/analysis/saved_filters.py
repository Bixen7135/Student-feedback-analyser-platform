"""Saved filter CRUD — persist and retrieve named filter configurations."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import orjson

from src.storage.database import Database
from src.utils.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("filter_config"), str):
        try:
            d["filter_config"] = orjson.loads(d["filter_config"])
        except Exception:
            d["filter_config"] = {}
    return d


def create_saved_filter(
    db: Database,
    name: str,
    entity_type: str,
    filter_config: dict,
) -> dict[str, Any]:
    """Create and persist a new saved filter. Returns the saved record."""
    filter_id = f"sf_{uuid.uuid4().hex[:16]}"
    now = _utcnow()
    db.execute(
        "INSERT INTO saved_filters (id, name, entity_type, filter_config, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            filter_id,
            name,
            entity_type,
            orjson.dumps(filter_config).decode(),
            now,
        ),
    )
    db.commit()
    log.info("saved_filter_created", filter_id=filter_id, name=name, entity_type=entity_type)
    row = db.fetchone("SELECT * FROM saved_filters WHERE id = ?", (filter_id,))
    return _row_to_dict(row)


def get_saved_filter(db: Database, filter_id: str) -> dict[str, Any] | None:
    """Fetch a single saved filter by ID."""
    row = db.fetchone("SELECT * FROM saved_filters WHERE id = ?", (filter_id,))
    return _row_to_dict(row) if row else None


def list_saved_filters(
    db: Database,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """List saved filters, optionally filtered by entity_type."""
    if entity_type:
        rows = db.fetchall(
            "SELECT * FROM saved_filters WHERE entity_type = ? ORDER BY created_at DESC",
            (entity_type,),
        )
    else:
        rows = db.fetchall("SELECT * FROM saved_filters ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]


def update_saved_filter(
    db: Database,
    filter_id: str,
    name: str | None = None,
    filter_config: dict | None = None,
) -> dict[str, Any] | None:
    """Update name and/or filter_config of a saved filter. Returns None if not found."""
    updates: list[str] = []
    params: list[Any] = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if filter_config is not None:
        updates.append("filter_config = ?")
        params.append(orjson.dumps(filter_config).decode())

    if not updates:
        return get_saved_filter(db, filter_id)

    params.append(filter_id)
    db.execute(
        f"UPDATE saved_filters SET {', '.join(updates)} WHERE id = ?",
        tuple(params),
    )
    db.commit()
    return get_saved_filter(db, filter_id)


def delete_saved_filter(db: Database, filter_id: str) -> bool:
    """Delete a saved filter. Returns True if deleted, False if not found."""
    result = db.execute("DELETE FROM saved_filters WHERE id = ?", (filter_id,))
    db.commit()
    return result.rowcount > 0
