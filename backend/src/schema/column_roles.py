"""Shared column normalization, aliasing, and role resolution."""
from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from src.ingest.loader import COLUMN_MAP
from src.schema.types import DatasetSchemaSnapshot, ResolvedColumns

TEXT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "text_processed",
    "text_feedback",
    "text",
    "feedback",
    "comment",
    "review",
    "response",
)

TASK_STANDARD_COLUMNS: dict[str, str] = {
    "language": "language",
    "sentiment": "sentiment_class",
    "detail_level": "detail_level",
}

ROLE_TO_STANDARD_COLUMN: dict[str, str] = {
    "text": "text_feedback",
    **TASK_STANDARD_COLUMNS,
}

# Conservative aliases used when explicit role metadata is missing or incomplete.
STANDARD_COLUMN_ALIASES: dict[str, set[str]] = {
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
        "тональностькласс",
        "класстональности",
    },
    "language": {
        "language",
        "язык",
    },
    "detail_level": {
        "detaillevel",
        "length",
        "длина",
    },
}


def normalize_column_name(name: str) -> str:
    """Apply the canonical header normalization used across the backend."""
    stripped = str(name).strip()
    return COLUMN_MAP.get(stripped, stripped)


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame headers with a single strip + COLUMN_MAP pass."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized != col:
            rename_map[col] = normalized
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def normalize_column_roles(column_roles: Mapping[str, str] | None) -> dict[str, str]:
    """Normalize role-mapping keys through the same header contract."""
    normalized: dict[str, str] = {}
    for col, role in (column_roles or {}).items():
        normalized[normalize_column_name(col)] = role
    return normalized


def propagate_column_roles(
    existing_roles: Mapping[str, str] | None,
    renames: Mapping[str, str],
) -> dict[str, str]:
    """Apply column renames to an existing column_roles mapping."""
    updated: dict[str, str] = {}
    for col, role in (existing_roles or {}).items():
        updated[renames.get(col, col)] = role
    return updated


def infer_column_roles(df: pd.DataFrame) -> dict[str, str]:
    """Infer an initial column_roles map from current columns."""
    resolved = resolve_roles(df=df, column_roles={}, overrides={})
    roles: dict[str, str] = {}
    if resolved.text_col and normalize_column_name(resolved.text_col) in TEXT_COLUMN_CANDIDATES:
        roles[resolved.text_col] = "text"
    for task, col in resolved.label_col_by_task.items():
        roles[col] = task
    return roles


def standardize_role_columns(
    df: pd.DataFrame,
    column_roles: Mapping[str, str] | None,
) -> pd.DataFrame:
    """
    Normalize headers and map user-renamed role columns back to canonical names.

    A conservative alias fallback is applied only when no explicit role mapping exists.
    """
    normalized_df = normalize_dataframe_columns(df)
    normalized_roles = normalize_column_roles(column_roles)

    role_renames: dict[str, str] = {}
    for col, role in normalized_roles.items():
        target = ROLE_TO_STANDARD_COLUMN.get(role)
        if (
            target
            and col in normalized_df.columns
            and col != target
            and target not in normalized_df.columns
        ):
            role_renames[col] = target

    if role_renames:
        normalized_df = normalized_df.rename(columns=role_renames)

    fallback_renames = _fallback_standard_renames(normalized_df)
    if fallback_renames:
        normalized_df = normalized_df.rename(columns=fallback_renames)

    return normalized_df


def resolve_roles(
    df: pd.DataFrame,
    column_roles: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedColumns:
    """Resolve semantic columns from explicit overrides, role metadata, and aliases."""
    normalized_roles = normalize_column_roles(column_roles)
    snapshot = _build_snapshot(df, normalized_roles)
    override_map = _normalize_overrides(overrides)

    text_col = _pick_text_col(df, snapshot, normalized_roles, override_map)
    label_col_by_task: dict[str, str] = {}
    for task in TASK_STANDARD_COLUMNS:
        label_col = _pick_label_col(task, snapshot, normalized_roles, override_map)
        if label_col is not None:
            label_col_by_task[task] = label_col

    return ResolvedColumns(
        text_col=text_col,
        label_col_by_task=label_col_by_task,
        language_col=label_col_by_task.get("language"),
        detail_col=label_col_by_task.get("detail_level"),
    )


def _build_snapshot(
    df: pd.DataFrame,
    column_roles: dict[str, str],
) -> DatasetSchemaSnapshot:
    return DatasetSchemaSnapshot(
        columns=tuple(str(col) for col in df.columns),
        normalized_columns=tuple(normalize_column_name(col) for col in df.columns),
        column_roles=dict(column_roles),
    )


def _normalize_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, str | dict[str, str]]:
    raw = dict(overrides or {})
    normalized: dict[str, str | dict[str, str]] = {}

    text_override = raw.get("text_col")
    if isinstance(text_override, str) and text_override.strip():
        normalized["text_col"] = normalize_column_name(text_override)

    label_map: dict[str, str] = {}
    explicit_label_map = raw.get("label_col_by_task")
    if isinstance(explicit_label_map, Mapping):
        for task, col in explicit_label_map.items():
            if isinstance(col, str) and str(task) in TASK_STANDARD_COLUMNS and col.strip():
                label_map[str(task)] = normalize_column_name(col)

    for task in TASK_STANDARD_COLUMNS:
        candidate = raw.get(task)
        if isinstance(candidate, str) and candidate.strip():
            label_map[task] = normalize_column_name(candidate)

    language_override = raw.get("language_col")
    if isinstance(language_override, str) and language_override.strip():
        label_map["language"] = normalize_column_name(language_override)

    detail_override = raw.get("detail_col")
    if isinstance(detail_override, str) and detail_override.strip():
        label_map["detail_level"] = normalize_column_name(detail_override)

    if label_map:
        normalized["label_col_by_task"] = label_map

    return normalized


def _pick_text_col(
    df: pd.DataFrame,
    snapshot: DatasetSchemaSnapshot,
    column_roles: dict[str, str],
    overrides: dict[str, str | dict[str, str]],
) -> str | None:
    explicit = overrides.get("text_col")
    if isinstance(explicit, str):
        resolved = _resolve_existing_column(snapshot, explicit)
        if resolved is not None:
            return resolved

    for col, role in column_roles.items():
        if role == "text":
            resolved = _resolve_existing_column(snapshot, col)
            if resolved is not None:
                return resolved

    for candidate in TEXT_COLUMN_CANDIDATES:
        resolved = _resolve_existing_column(snapshot, candidate)
        if resolved is not None:
            return resolved

    for col in snapshot.columns:
        series = df[col]
        if series.dtype == object or series.dtype.kind in ("O", "U", "S"):
            return col
    return None


def _pick_label_col(
    task: str,
    snapshot: DatasetSchemaSnapshot,
    column_roles: dict[str, str],
    overrides: dict[str, str | dict[str, str]],
) -> str | None:
    explicit_map = overrides.get("label_col_by_task")
    if isinstance(explicit_map, dict):
        explicit = explicit_map.get(task)
        if explicit:
            resolved = _resolve_existing_column(snapshot, explicit)
            if resolved is not None:
                return resolved

    for col, role in column_roles.items():
        if role == task:
            resolved = _resolve_existing_column(snapshot, col)
            if resolved is not None:
                return resolved

    standard = TASK_STANDARD_COLUMNS[task]
    resolved_standard = _resolve_existing_column(snapshot, standard)
    if resolved_standard is not None:
        return resolved_standard

    alias_match = _find_alias_match(snapshot, standard)
    if alias_match is not None:
        return alias_match

    return None


def _resolve_existing_column(
    snapshot: DatasetSchemaSnapshot,
    target: str,
) -> str | None:
    for actual in snapshot.columns:
        if actual == target:
            return actual

    matches = [
        actual
        for actual, normalized in zip(snapshot.columns, snapshot.normalized_columns, strict=False)
        if normalized == target
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _find_alias_match(
    snapshot: DatasetSchemaSnapshot,
    standard: str,
) -> str | None:
    aliases = STANDARD_COLUMN_ALIASES.get(standard, set())
    if not aliases:
        return None

    matches: list[str] = []
    for actual, normalized in zip(snapshot.columns, snapshot.normalized_columns, strict=False):
        compact = _compact_col_name(normalized)
        if compact in aliases:
            matches.append(actual)

    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _fallback_standard_renames(df: pd.DataFrame) -> dict[str, str]:
    renames: dict[str, str] = {}
    snapshot = _build_snapshot(df, {})
    for standard in STANDARD_COLUMN_ALIASES:
        if standard in snapshot.columns:
            continue
        match = _find_alias_match(snapshot, standard)
        if match is not None and match != standard:
            renames[match] = standard
    return renames


def _compact_col_name(name: str) -> str:
    return re.sub(r"[\W_]+", "", str(name).strip().lower(), flags=re.UNICODE)
