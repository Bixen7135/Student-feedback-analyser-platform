"""Training configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.training.contract import (
    CLASSIFICATION_LOSS,
    validate_training_config_compatibility,
)


@dataclass
class TrainingConfig:
    """Configuration for a single user-initiated training run."""

    # Canonical classification contract field. Stored configs may round-trip it
    # back into TrainingConfig when rehydrating pipeline metadata.
    loss: str = CLASSIFICATION_LOSS

    # Split ratios must sum to 1.0
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Class balancing strategy
    # "none"         -> no balancing (raw class frequencies)
    # "class_weight" -> use class_weight='balanced' in LogisticRegression
    # "oversample"   -> random oversample minority classes to match majority
    class_balancing: str = "class_weight"

    # Optional baseline classifier hyperparameter overrides
    max_features: int | None = None
    C: float | None = None
    max_iter: int | None = None

    # Transformer fine-tuning defaults for xlm_roberta mode
    pretrained_model: str = "xlm-roberta-base"
    max_seq_length: int = 256
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    # Optional transformer knobs
    warmup_ratio: float | None = None
    gradient_accumulation_steps: int | None = None
    head_hidden_units: int | None = None
    dropout: float | None = None
    activation: str | None = None

    # Optional column overrides
    text_col: str | None = None
    label_col: str | None = None

    # Fine-tuning: ID of a previously-trained model to warm-start from
    base_model_id: str | None = None

    def to_dict(self, model_type: str | None = None) -> dict[str, Any]:
        raw = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "class_balancing": self.class_balancing,
            "max_features": self.max_features,
            "C": self.C,
            "max_iter": self.max_iter,
            "pretrained_model": self.pretrained_model,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "head_hidden_units": self.head_hidden_units,
            "dropout": self.dropout,
            "activation": self.activation,
            "text_col": self.text_col,
            "label_col": self.label_col,
            "base_model_id": self.base_model_id,
        }
        if model_type is None:
            return raw
        return validate_training_config_compatibility(
            model_type=model_type,
            config=raw,
        )
