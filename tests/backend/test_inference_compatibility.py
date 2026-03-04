"""Unit tests for the shared inference compatibility contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.inference.engine import check_compatibility, run_inference
from src.schema import resolve_roles
from src.storage.models import ModelMeta


def _make_model_meta(
    *,
    model_type: str = "tfidf",
    storage_path: str = "C:/tmp/model_test_compat",
    preprocess_spec_id: str = "preprocess_v1",
) -> ModelMeta:
    return ModelMeta(
        id="model_test_compat",
        name="Compatibility Test Model",
        task="sentiment",
        model_type=model_type,
        version=1,
        created_at="2026-02-28T00:00:00+00:00",
        storage_path=storage_path,
        input_signature={
            "task": "sentiment",
            "required_roles": ["text"],
            "text": {
                "role": "text",
                "source_column": "text_feedback",
                "model_input_column": "text_model_input",
            },
            "label_schema": {
                "role": "sentiment",
                "column": "sentiment_class",
                "class_order": ["negative", "neutral", "positive"],
            },
            "preprocess_spec_id": preprocess_spec_id,
        },
        preprocess_spec={"id": preprocess_spec_id},
        metrics={"classes": ["negative", "neutral", "positive"]},
    )


def test_check_compatibility_ok_for_matching_schema() -> None:
    df = pd.DataFrame(
        {
            "text_feedback": ["good", "bad"],
            "sentiment_class": ["positive", "negative"],
        }
    )
    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})

    report = check_compatibility(
        model_meta=_make_model_meta(),
        dataset_schema=df,
        resolved_columns=resolved_columns,
    )

    assert report["ok"] is True
    assert report["reasons"] == []
    assert report["preprocess_spec_id"] == "preprocess_v1"
    assert report["text_col_used"] == "text_feedback"
    assert report["label_schema"]["class_order"] == [
        "negative",
        "neutral",
        "positive",
    ]


def test_check_compatibility_fails_without_text_role() -> None:
    df = pd.DataFrame({"q1": [1, 2], "q2": [3, 4]})
    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})

    report = check_compatibility(
        model_meta=_make_model_meta(),
        dataset_schema=df,
        resolved_columns=resolved_columns,
    )

    assert report["ok"] is False
    assert report["reasons"][0]["code"] == "missing_required_role"
    assert report["text_col_used"] is None


def test_check_compatibility_fails_for_unsupported_preprocess_spec() -> None:
    df = pd.DataFrame(
        {
            "text_feedback": ["good", "bad"],
            "sentiment_class": ["positive", "negative"],
        }
    )
    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})

    report = check_compatibility(
        model_meta=_make_model_meta(preprocess_spec_id="preprocess_v2"),
        dataset_schema=df,
        resolved_columns=resolved_columns,
    )

    assert report["ok"] is False
    assert any(
        reason["code"] == "unsupported_preprocess_spec"
        for reason in report["reasons"]
    )


def test_run_inference_resolves_directory_artifacts_before_model_dispatch(tmp_path) -> None:
    model_dir = tmp_path / "registered_model" / "v1" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        (
            '{"pretrained_model":"xlm-roberta-base","max_seq_length":64,'
            '"batch_size":2,"epochs":1,"learning_rate":0.001,"weight_decay":0.0,'
            '"dropout":0.1,"activation":"gelu","classes":["negative","positive"],'
            '"label_to_id":{"negative":0,"positive":1}}'
        ),
        encoding="utf-8",
    )

    df = pd.DataFrame({"text_feedback": ["good", "bad"]})
    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})

    with pytest.raises(RuntimeError, match="sfap-backend\\[transformers\\]"):
        run_inference(
            df=df,
            model_meta=_make_model_meta(
                model_type="xlm_roberta",
                storage_path=str(model_dir.parent),
            ),
            resolved_columns=resolved_columns,
        )
