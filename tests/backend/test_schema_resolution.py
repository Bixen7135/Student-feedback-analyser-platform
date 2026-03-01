"""Unit tests for shared schema resolution helpers."""
from __future__ import annotations

import pandas as pd

from src.schema import (
    infer_column_roles,
    resolve_roles,
    standardize_role_columns,
)


def test_resolve_roles_prefers_column_roles_and_alias_fallback() -> None:
    df = pd.DataFrame(
        {
            "free_text": ["a", "b"],
            "tone_label": ["positive", "negative"],
            "length": ["short", "long"],
        }
    )

    resolved = resolve_roles(
        df=df,
        column_roles={"free_text": "text", "tone_label": "sentiment"},
        overrides={},
    )

    assert resolved.text_col == "free_text"
    assert resolved.label_col_by_task["sentiment"] == "tone_label"
    assert resolved.detail_col == "length"
    assert resolved.language_col is None


def test_resolve_roles_normalizes_override_names() -> None:
    df = pd.DataFrame(
        {
            " comment ": ["good", "bad"],
            "sentiment_class": ["positive", "negative"],
        }
    )

    resolved = resolve_roles(
        df=df,
        column_roles={},
        overrides={"text_col": "comment"},
    )

    assert resolved.text_col == " comment "
    assert resolved.label_col_by_task["sentiment"] == "sentiment_class"


def test_standardize_role_columns_keeps_ambiguous_aliases_unmapped() -> None:
    df = pd.DataFrame(
        {
            "text": ["a"],
            "feedback": ["b"],
            "sentiment": ["positive"],
        }
    )

    standardized = standardize_role_columns(df, column_roles={})

    assert "text_feedback" not in standardized.columns
    assert "text" in standardized.columns
    assert "feedback" in standardized.columns
    assert "sentiment_class" in standardized.columns


def test_infer_column_roles_uses_canonical_resolution() -> None:
    df = pd.DataFrame(
        {
            "text_processed": ["a"],
            "language": ["ru"],
            "detail_level": ["short"],
        }
    )

    roles = infer_column_roles(df)

    assert roles == {
        "text_processed": "text",
        "language": "language",
        "detail_level": "detail_level",
    }
