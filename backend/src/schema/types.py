"""Shared value types for schema normalization and column-role resolution."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResolvedColumns:
    """Resolved semantic columns for text and task labels."""

    text_col: str | None = None
    label_col_by_task: dict[str, str] = field(default_factory=dict)
    language_col: str | None = None
    detail_col: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "text_col": self.text_col,
            "label_col_by_task": dict(self.label_col_by_task),
            "language_col": self.language_col,
            "detail_col": self.detail_col,
        }


@dataclass(frozen=True)
class DatasetSchemaSnapshot:
    """Normalized view of dataset columns and stored role assignments."""

    columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    column_roles: dict[str, str] = field(default_factory=dict)
