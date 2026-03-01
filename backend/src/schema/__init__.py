"""Shared schema helpers for dataset column normalization and role resolution."""

from src.schema.column_roles import (
    ROLE_TO_STANDARD_COLUMN,
    TASK_STANDARD_COLUMNS,
    TEXT_COLUMN_CANDIDATES,
    infer_column_roles,
    normalize_column_name,
    normalize_column_roles,
    normalize_dataframe_columns,
    propagate_column_roles,
    resolve_roles,
    standardize_role_columns,
)
from src.schema.types import DatasetSchemaSnapshot, ResolvedColumns

__all__ = [
    "DatasetSchemaSnapshot",
    "ResolvedColumns",
    "ROLE_TO_STANDARD_COLUMN",
    "TASK_STANDARD_COLUMNS",
    "TEXT_COLUMN_CANDIDATES",
    "infer_column_roles",
    "normalize_column_name",
    "normalize_column_roles",
    "normalize_dataframe_columns",
    "propagate_column_roles",
    "resolve_roles",
    "standardize_role_columns",
]
