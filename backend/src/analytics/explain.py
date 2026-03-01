"""Local explanation helpers for linear text models."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.feature_importance import load_linear_text_components
from src.inference.signature import canonicalize_signature
from src.preprocessing.spec import DEFAULT_PREPROCESS_SPEC, apply_preprocess
from src.storage.models import ModelMeta


def explain_text_instance(
    model_meta: ModelMeta,
    text: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return top contributing features for a single text input."""
    components = load_linear_text_components(model_meta)
    pipeline = components["pipeline"]
    vectorizer = components["vectorizer"]
    feature_names = components["feature_names"]
    coef_matrix = components["coef_matrix"]
    classes = components["classes"]

    processed_text = _prepare_text_for_model(model_meta, text)
    pred = pipeline.predict([processed_text])[0]
    pred_label = str(pred)
    if pred_label not in classes:
        pred_idx = int(np.argmax(pipeline.predict_proba([processed_text])[0]))
        pred_label = classes[pred_idx]
    else:
        pred_idx = classes.index(pred_label)

    transformed = vectorizer.transform([processed_text])
    row = transformed.getrow(0)
    indices = row.indices
    values = row.data

    contributions: list[dict[str, Any]] = []
    if len(indices) > 0:
        class_weights = coef_matrix[pred_idx]
        raw_contrib = values * class_weights[indices]
        order = np.argsort(np.abs(raw_contrib))[::-1][:top_n]
        for pos in order:
            feat_idx = int(indices[int(pos)])
            contributions.append(
                {
                    "feature": feature_names[feat_idx],
                    "value": round(float(values[int(pos)]), 6),
                    "weight": round(float(class_weights[feat_idx]), 6),
                    "contribution": round(float(raw_contrib[int(pos)]), 6),
                }
            )

    probabilities = pipeline.predict_proba([processed_text])[0]
    return {
        "model_id": model_meta.id,
        "predicted_class": pred_label,
        "probabilities": {
            class_name: round(float(prob), 6)
            for class_name, prob in zip(classes, probabilities, strict=False)
        },
        "processed_text": processed_text,
        "top_features": contributions,
    }


def _prepare_text_for_model(model_meta: ModelMeta, text: str) -> str:
    signature = canonicalize_signature(model_meta)
    preprocess_spec_id = signature.get("preprocess_spec_id")
    raw_text = str(text or "")
    if preprocess_spec_id == DEFAULT_PREPROCESS_SPEC.id:
        df = pd.DataFrame({"text_feedback": [raw_text]})
        processed_df = apply_preprocess(df, text_col="text_feedback", spec=DEFAULT_PREPROCESS_SPEC)
        return str(processed_df["text_processed"].iloc[0])
    return raw_text
