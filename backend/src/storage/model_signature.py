"""Helpers for model input signatures and lightweight training profiles."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.spec import DEFAULT_PREPROCESS_SPEC, PreprocessSpec
from src.schema.types import ResolvedColumns
from src.text_tasks.base import TextClassifier


def build_input_signature(
    *,
    task: str,
    resolved_columns: ResolvedColumns,
    source_text_col: str,
    model_input_col: str,
    label_col: str,
    preprocess_spec: PreprocessSpec | None = None,
    training_config: dict[str, Any] | None = None,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    resolved_spec = preprocess_spec or DEFAULT_PREPROCESS_SPEC
    return {
        "task": task,
        "required_roles": ["text"],
        "text": {
            "role": "text",
            "source_column": source_text_col,
            "model_input_column": model_input_col,
        },
        "label_schema": {
            "role": task,
            "column": label_col,
            "class_order": [str(value) for value in (classes or [])],
        },
        "resolved_columns": resolved_columns.as_dict(),
        "preprocess_spec_id": resolved_spec.id,
        "training_config": dict(training_config or {}),
    }


def build_training_profile(
    *,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task: str,
    clf: TextClassifier,
) -> dict[str, Any]:
    text_series = df[text_col].fillna("").astype(str)
    char_lengths = text_series.str.len().to_numpy(dtype=float)
    word_lengths = text_series.str.split().str.len().to_numpy(dtype=float)
    labels = df[label_col].fillna("").astype(str)
    class_priors = labels.value_counts(normalize=True).to_dict()

    return {
        "task": task,
        "n_rows": int(len(df)),
        "text_model_input_column": text_col,
        "text_length_chars": _quantiles(char_lengths),
        "text_length_words": _quantiles(word_lengths),
        "class_priors": {
            str(label): round(float(prior), 4)
            for label, prior in class_priors.items()
        },
        "vocabulary_size": _vocabulary_size(clf),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("p05", "p25", "p50", "p75", "p95")}
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p05": round(float(q[0]), 2),
        "p25": round(float(q[1]), 2),
        "p50": round(float(q[2]), 2),
        "p75": round(float(q[3]), 2),
        "p95": round(float(q[4]), 2),
    }


def _vocabulary_size(clf: TextClassifier) -> int:
    pipeline = getattr(clf, "_pipeline", None)
    if pipeline is None:
        return 0
    named_steps = getattr(pipeline, "named_steps", {})
    vectorizer = named_steps.get("tfidf")
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    if isinstance(vocabulary, dict):
        return int(len(vocabulary))
    return 0
