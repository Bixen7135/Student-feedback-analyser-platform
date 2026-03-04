"""Inference execution and compatibility checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.signature import canonicalize_signature
from src.preprocessing.spec import DEFAULT_PREPROCESS_SPEC, apply_preprocess
from src.schema import TEXT_COLUMN_CANDIDATES, normalize_column_name
from src.schema.types import DatasetSchemaSnapshot, ResolvedColumns
from src.storage.model_registry import resolve_model_artifact_path
from src.storage.models import ModelMeta
from src.text_tasks.base import TextClassifier
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.xlm_roberta_classifier import XlmRobertaClassifier
from src.training.contract import MODEL_TYPE_XLM_ROBERTA


def check_compatibility(
    model_meta: ModelMeta,
    dataset_schema: Any,
    resolved_columns: ResolvedColumns,
) -> dict[str, Any]:
    """Validate whether a model can run against the given dataset mapping."""
    signature = canonicalize_signature(model_meta)
    schema_columns, schema_roles = _schema_contract(dataset_schema)
    reasons: list[dict[str, Any]] = []

    required_roles = set(signature["required_roles"])
    if "text" in required_roles and not resolved_columns.text_col:
        reasons.append(
            {
                "code": "missing_required_role",
                "role": "text",
                "message": "No text column could be resolved for this dataset.",
                "suggested_fix": "Assign a column role 'text' or choose a valid text column.",
            }
        )
    elif resolved_columns.text_col and resolved_columns.text_col not in schema_columns:
        reasons.append(
            {
                "code": "missing_column",
                "role": "text",
                "column": resolved_columns.text_col,
                "message": f"Resolved text column '{resolved_columns.text_col}' is not present in the dataset schema.",
                "suggested_fix": "Refresh the dataset schema or assign the correct text column role.",
            }
        )
    elif "text" in required_roles and resolved_columns.text_col:
        if not _is_supported_text_mapping(
            resolved_text_col=resolved_columns.text_col,
            schema_roles=schema_roles,
            signature=signature,
        ):
            reasons.append(
                {
                    "code": "missing_required_role",
                    "role": "text",
                    "column": resolved_columns.text_col,
                    "message": (
                        f"Resolved text column '{resolved_columns.text_col}' only matched a "
                        "generic fallback and is not an assigned or canonical text field."
                    ),
                    "suggested_fix": (
                        "Assign the dataset's text column role explicitly or choose the correct "
                        "text column before running analysis."
                    ),
                }
            )

    preprocess_spec_id = signature.get("preprocess_spec_id")
    if preprocess_spec_id and preprocess_spec_id != DEFAULT_PREPROCESS_SPEC.id:
        reasons.append(
            {
                "code": "unsupported_preprocess_spec",
                "preprocess_spec_id": preprocess_spec_id,
                "message": f"Preprocess spec '{preprocess_spec_id}' is not supported by the current inference engine.",
                "suggested_fix": "Retrain or re-register the model with a supported preprocess spec.",
            }
        )

    label_role = signature["label_schema"].get("role")
    resolved_label = resolved_columns.label_col_by_task.get(str(label_role)) if label_role else None

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "resolved_columns": resolved_columns.as_dict(),
        "required_roles": list(signature["required_roles"]),
        "preprocess_spec_id": preprocess_spec_id,
        "label_schema": signature["label_schema"],
        "schema_columns": schema_columns,
        "text_col_used": resolved_columns.text_col,
        "label_col_used": resolved_label,
    }


def run_inference(
    df: pd.DataFrame,
    model_meta: ModelMeta,
    resolved_columns: ResolvedColumns,
) -> dict[str, Any]:
    """Apply the model's preprocessing contract and return predictions."""
    compatibility_roles = (
        {resolved_columns.text_col: "text"}
        if resolved_columns.text_col
        else {}
    )
    dataset_schema = DatasetSchemaSnapshot(
        columns=tuple(str(col) for col in df.columns),
        normalized_columns=tuple(normalize_column_name(col) for col in df.columns),
        column_roles=compatibility_roles,
    )
    compatibility = check_compatibility(
        model_meta=model_meta,
        dataset_schema=dataset_schema,
        resolved_columns=resolved_columns,
    )
    if not compatibility["ok"]:
        raise ValueError(_compatibility_message(compatibility))

    signature = canonicalize_signature(model_meta)
    source_text_col = resolved_columns.text_col
    assert source_text_col is not None, "Compatibility check must guarantee a text column"

    working_df = df.copy()
    preprocess_spec_id = signature.get("preprocess_spec_id")
    model_input_col = str(
        signature["text"].get("model_input_column")
        or source_text_col
    )

    if preprocess_spec_id == DEFAULT_PREPROCESS_SPEC.id:
        working_df = apply_preprocess(working_df, text_col=source_text_col, spec=DEFAULT_PREPROCESS_SPEC)
        if model_input_col != "text_processed":
            working_df[model_input_col] = working_df["text_processed"]
    else:
        if model_input_col != source_text_col:
            working_df[model_input_col] = working_df[source_text_col].fillna("").astype(str)

    artifact_path = resolve_model_artifact_path(model_meta.storage_path)
    clf = _load_classifier(artifact_path, model_meta.model_type)
    texts = working_df[model_input_col].fillna("").astype(str).tolist()
    preds = clf.predict(texts)

    try:
        probas = clf.predict_proba(texts)
        class_list = clf.classes_
        conf_vals = np.array(
            [probas[i, class_list.index(preds[i])] for i in range(len(preds))]
        )
    except Exception:
        conf_vals = np.full(len(preds), float("nan"))

    pred_series = pd.Series(preds)
    counts = pred_series.value_counts()
    class_dist = {str(k): round(float(v) / len(preds), 4) for k, v in counts.items()}

    return {
        "predictions": preds,
        "confidences": conf_vals,
        "classes": clf.classes_,
        "n_predicted": int((~pred_series.isna()).sum()),
        "class_distribution": class_dist,
        "compatibility": compatibility,
        "preprocess_spec_applied": preprocess_spec_id,
        "text_col_used": source_text_col,
        "model_input_col": model_input_col,
    }


def _load_classifier(model_path: Path, model_type: str) -> TextClassifier:
    if model_type == "tfidf":
        return TfidfClassifier.load(model_path)
    if model_type == "char_ngram":
        return CharNgramClassifier.load(model_path)
    if model_type == MODEL_TYPE_XLM_ROBERTA:
        return XlmRobertaClassifier.load(model_path)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _schema_contract(dataset_schema: Any) -> tuple[list[str], dict[str, str]]:
    if isinstance(dataset_schema, DatasetSchemaSnapshot):
        return list(dataset_schema.columns), dict(dataset_schema.column_roles)
    if isinstance(dataset_schema, dict):
        columns = dataset_schema.get("columns")
        column_roles = dataset_schema.get("column_roles")
        if isinstance(columns, list):
            return [str(col) for col in columns], _normalize_roles(column_roles)
    if isinstance(dataset_schema, pd.DataFrame):
        return [str(col) for col in dataset_schema.columns], {}
    if isinstance(dataset_schema, list):
        cols: list[str] = []
        for item in dataset_schema:
            if isinstance(item, dict) and "name" in item:
                cols.append(str(item["name"]))
            elif hasattr(item, "name"):
                cols.append(str(getattr(item, "name")))
        return cols, {}
    return [], {}


def _is_supported_text_mapping(
    *,
    resolved_text_col: str,
    schema_roles: dict[str, str],
    signature: dict[str, Any],
) -> bool:
    normalized_text = normalize_column_name(resolved_text_col)
    if any(
        role == "text" and normalize_column_name(col) == normalized_text
        for col, role in schema_roles.items()
    ):
        return True

    allowed_text_names = {normalize_column_name(col) for col in TEXT_COLUMN_CANDIDATES}
    source_column = signature["text"].get("source_column")
    if isinstance(source_column, str) and source_column.strip():
        allowed_text_names.add(normalize_column_name(source_column))

    return normalized_text in allowed_text_names


def _normalize_roles(raw_roles: Any) -> dict[str, str]:
    if not isinstance(raw_roles, dict):
        return {}
    return {
        str(col): str(role)
        for col, role in raw_roles.items()
        if str(col) and str(role)
    }


def _compatibility_message(report: dict[str, Any]) -> str:
    reasons = report.get("reasons") or []
    if reasons:
        return "; ".join(str(reason.get("message") or reason.get("code") or "incompatible") for reason in reasons)
    return "Model is incompatible with the target dataset."
