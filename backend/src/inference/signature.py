"""Canonical schema helpers for model inference signatures."""
from __future__ import annotations

from typing import Any

from src.storage.models import ModelMeta


def canonicalize_signature(model_meta: ModelMeta) -> dict[str, Any]:
    """Normalize persisted signature fields into a stable inference contract."""
    input_signature = dict(model_meta.input_signature or {})
    text_info = dict(input_signature.get("text") or {})

    label_info = input_signature.get("label_schema")
    if not isinstance(label_info, dict):
        label_info = dict(input_signature.get("label") or {})

    classes = label_info.get("class_order")
    if not isinstance(classes, list):
        metrics_classes = model_meta.metrics.get("classes") if isinstance(model_meta.metrics, dict) else None
        classes = list(metrics_classes) if isinstance(metrics_classes, list) else []

    preprocess_spec = dict(model_meta.preprocess_spec or {})
    preprocess_spec_id = (
        input_signature.get("preprocess_spec_id")
        or preprocess_spec.get("id")
        or None
    )

    required_roles = input_signature.get("required_roles")
    if not isinstance(required_roles, list) or not required_roles:
        required_roles = ["text"]

    return {
        "task": str(input_signature.get("task") or model_meta.task),
        "required_roles": [str(role) for role in required_roles if str(role)],
        "preprocess_spec_id": str(preprocess_spec_id) if preprocess_spec_id else None,
        "text": {
            "role": str(text_info.get("role") or "text"),
            "source_column": text_info.get("source_column"),
            "model_input_column": text_info.get("model_input_column") or text_info.get("source_column"),
        },
        "label_schema": {
            "role": str(label_info.get("role") or model_meta.task),
            "column": label_info.get("column"),
            "class_order": [str(value) for value in classes],
        },
        "resolved_columns": dict(input_signature.get("resolved_columns") or {}),
        "training_config": dict(input_signature.get("training_config") or {}),
    }
