"""Canonical training parameter contract for user-initiated model training."""
from __future__ import annotations

from typing import Any, Mapping

MODEL_TYPE_TFIDF = "tfidf"
MODEL_TYPE_CHAR_NGRAM = "char_ngram"
MODEL_TYPE_XLM_ROBERTA = "xlm_roberta"

BASELINE_MODEL_TYPES = frozenset({
    MODEL_TYPE_TFIDF,
    MODEL_TYPE_CHAR_NGRAM,
})
VALID_MODEL_TYPES = frozenset({
    MODEL_TYPE_TFIDF,
    MODEL_TYPE_CHAR_NGRAM,
    MODEL_TYPE_XLM_ROBERTA,
})

CLASSIFICATION_LOSS = "cross_entropy"

COMMON_CONFIG_FIELDS = (
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "class_balancing",
    "text_col",
    "label_col",
    "base_model_id",
)

BASELINE_ONLY_FIELDS = (
    "max_features",
    "C",
    "max_iter",
)

TRANSFORMER_REQUIRED_FIELDS = (
    "pretrained_model",
    "max_seq_length",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
)

TRANSFORMER_OPTIONAL_FIELDS = (
    "warmup_ratio",
    "gradient_accumulation_steps",
    "head_hidden_units",
    "dropout",
    "activation",
)

XLM_ROBERTA_DEFAULTS: dict[str, Any] = {
    "pretrained_model": "xlm-roberta-base",
    "max_seq_length": 256,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
}

# The canonical parameter table is used as the source of truth for validation
# and serialization. The UI can also consume it later through the API if needed.
TRAINING_PARAMETER_TABLE: tuple[dict[str, Any], ...] = (
    {"name": "loss", "applies_to": "all", "required": True, "default": CLASSIFICATION_LOSS},
    {"name": "train_ratio", "applies_to": "all", "required": True, "default": 0.80},
    {"name": "val_ratio", "applies_to": "all", "required": True, "default": 0.10},
    {"name": "test_ratio", "applies_to": "all", "required": True, "default": 0.10},
    {"name": "class_balancing", "applies_to": "all", "required": True, "default": "class_weight"},
    {"name": "text_col", "applies_to": "all", "required": False, "default": None},
    {"name": "label_col", "applies_to": "all", "required": False, "default": None},
    {
        "name": "max_features",
        "applies_to": [MODEL_TYPE_TFIDF, MODEL_TYPE_CHAR_NGRAM],
        "required": False,
        "default": None,
    },
    {
        "name": "C",
        "applies_to": [MODEL_TYPE_TFIDF, MODEL_TYPE_CHAR_NGRAM],
        "required": False,
        "default": None,
    },
    {
        "name": "max_iter",
        "applies_to": [MODEL_TYPE_TFIDF, MODEL_TYPE_CHAR_NGRAM],
        "required": False,
        "default": None,
    },
    {
        "name": "pretrained_model",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["pretrained_model"],
    },
    {
        "name": "max_seq_length",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["max_seq_length"],
    },
    {
        "name": "batch_size",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["batch_size"],
    },
    {
        "name": "epochs",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["epochs"],
    },
    {
        "name": "learning_rate",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["learning_rate"],
    },
    {
        "name": "weight_decay",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": True,
        "default": XLM_ROBERTA_DEFAULTS["weight_decay"],
    },
    {
        "name": "warmup_ratio",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": False,
        "default": None,
    },
    {
        "name": "gradient_accumulation_steps",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": False,
        "default": None,
    },
    {
        "name": "head_hidden_units",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": False,
        "default": None,
    },
    {
        "name": "dropout",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": False,
        "default": None,
    },
    {
        "name": "activation",
        "applies_to": [MODEL_TYPE_XLM_ROBERTA],
        "required": False,
        "default": None,
    },
    {"name": "base_model_id", "applies_to": "all", "required": False, "default": None},
)


def config_fields_for_model_type(model_type: str) -> tuple[str, ...]:
    """Return the serializable config field list for a given model type."""
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Valid: {sorted(VALID_MODEL_TYPES)}"
        )

    fields = list(COMMON_CONFIG_FIELDS)
    if model_type in BASELINE_MODEL_TYPES:
        fields.extend(BASELINE_ONLY_FIELDS)
    elif model_type == MODEL_TYPE_XLM_ROBERTA:
        fields.extend(TRANSFORMER_REQUIRED_FIELDS)
        fields.extend(TRANSFORMER_OPTIONAL_FIELDS)
    return tuple(fields)


def canonicalize_training_config(
    model_type: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Filter config to the selected model type and inject canonical defaults."""
    fields = config_fields_for_model_type(model_type)
    raw = dict(config or {})

    canonical: dict[str, Any] = {"loss": CLASSIFICATION_LOSS}
    for field in COMMON_CONFIG_FIELDS:
        canonical[field] = raw.get(field)

    if model_type in BASELINE_MODEL_TYPES:
        for field in BASELINE_ONLY_FIELDS:
            canonical[field] = raw.get(field)
        return canonical

    for field in TRANSFORMER_REQUIRED_FIELDS:
        value = raw.get(field)
        if value is None:
            value = XLM_ROBERTA_DEFAULTS[field]
        canonical[field] = value

    for field in TRANSFORMER_OPTIONAL_FIELDS:
        canonical[field] = raw.get(field)

    return canonical


def validate_training_config_compatibility(
    model_type: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return canonical config and ensure required model-specific fields exist."""
    canonical = canonicalize_training_config(model_type=model_type, config=config)

    if model_type == MODEL_TYPE_XLM_ROBERTA:
        missing = [
            field
            for field in TRANSFORMER_REQUIRED_FIELDS
            if canonical.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(
                "xlm_roberta requires the following config fields: "
                + ", ".join(missing)
            )

    return canonical
