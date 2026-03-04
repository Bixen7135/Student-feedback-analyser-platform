"""Focused tests for XLM-RoBERTa inference and analytics compatibility."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.analytics.embeddings_service import compute_or_load_embeddings
from src.analytics.explain import explain_text_instance
from src.analytics.feature_importance import get_global_feature_importance
from src.inference.engine import run_inference
from src.schema import resolve_roles
from src.storage.models import ModelMeta


class _FakeTokenizer:
    def __init__(self, source: str) -> None:
        self.source = source

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.txt").write_text(self.source, encoding="utf-8")


class _FakeEncoderConfig:
    hidden_size = 6


class _FakeEncoder:
    def __init__(self, source: str) -> None:
        self.source = source
        self.config = _FakeEncoderConfig()

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "encoder.txt").write_text(self.source, encoding="utf-8")


class _FakeNetwork:
    def __init__(self, encoder: _FakeEncoder, num_labels: int) -> None:
        self.encoder = encoder
        self.classifier = self
        self.num_labels = num_labels
        self.class_bias = np.zeros(num_labels, dtype=float)

    def state_dict(self) -> dict[str, list[float]]:
        return {"class_bias": self.class_bias.tolist()}

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        self.class_bias = np.array(state["class_bias"], dtype=float)


class _FakeXlmBackend:
    def create_tokenizer(self, source: str) -> _FakeTokenizer:
        return _FakeTokenizer(source)

    def create_encoder(self, source: str) -> _FakeEncoder:
        return _FakeEncoder(source)

    def build_network(
        self,
        *,
        encoder: _FakeEncoder,
        num_labels: int,
        head_hidden_units: int | None,
        dropout: float,
        activation: str,
    ) -> _FakeNetwork:
        return _FakeNetwork(encoder, num_labels)

    def train(
        self,
        *,
        network: _FakeNetwork,
        tokenizer: _FakeTokenizer,
        texts: list[str],
        label_ids: list[int],
        max_seq_length: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float | None,
        gradient_accumulation_steps: int | None,
        seed: int,
    ) -> None:
        totals = np.zeros(network.num_labels, dtype=float)
        counts = np.zeros(network.num_labels, dtype=float)
        for text, label_id in zip(texts, label_ids, strict=False):
            totals[int(label_id)] += float(len(text))
            counts[int(label_id)] += 1.0
        network.class_bias = totals / np.maximum(counts, 1.0)

    def predict_logits(
        self,
        *,
        network: _FakeNetwork,
        tokenizer: _FakeTokenizer,
        texts: list[str],
        max_seq_length: int,
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        scale = np.linspace(0.01, 0.05, network.num_labels)
        return np.vstack([network.class_bias + (len(text) * scale) for text in texts])

    def embed_vectors(
        self,
        *,
        network: _FakeNetwork,
        tokenizer: _FakeTokenizer,
        texts: list[str],
        max_seq_length: int,
        batch_size: int,
        pooling: str,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, network.encoder.config.hidden_size), dtype=float)
        offset = 1.0 if pooling == "cls" else 0.5
        width = network.encoder.config.hidden_size
        return np.vstack(
            [
                np.linspace(
                    len(text) + offset,
                    len(text) + offset + width - 1,
                    width,
                )
                for text in texts
            ]
        )

    def save_pretrained(self, *, network: _FakeNetwork, tokenizer: _FakeTokenizer, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        network.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "tokenizer")
        (path / "head.pt").write_text(json.dumps(network.state_dict()), encoding="utf-8")

    def load_weights(self, *, network: _FakeNetwork, path: Path) -> None:
        payload = json.loads((path / "head.pt").read_text(encoding="utf-8"))
        network.load_state_dict(payload)

    def copy_state(self, *, source: _FakeNetwork, target: _FakeNetwork) -> None:
        target.load_state_dict(source.state_dict())


def _make_model_meta(*, storage_path: Path, model_id: str = "model_xlm_test") -> ModelMeta:
    return ModelMeta(
        id=model_id,
        name="XLM Analytics Test Model",
        task="sentiment",
        model_type="xlm_roberta",
        version=1,
        created_at="2026-03-01T00:00:00+00:00",
        storage_path=str(storage_path),
        metrics={"classes": ["negative", "positive"]},
        preprocess_spec={"id": "preprocess_v1"},
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
                "class_order": ["negative", "positive"],
            },
            "preprocess_spec_id": "preprocess_v1",
        },
    )


def _save_stub_xlm_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelMeta:
    from src.text_tasks.xlm_roberta_classifier import XlmRobertaClassifier

    monkeypatch.setattr(
        XlmRobertaClassifier,
        "_backend_factory",
        staticmethod(lambda: _FakeXlmBackend()),
    )

    clf = XlmRobertaClassifier(
        pretrained_model="xlm-roberta-base",
        max_seq_length=64,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        head_hidden_units=8,
        dropout=0.2,
        activation="relu",
    )
    clf.fit(
        ["excellent course", "very helpful teacher", "bad pacing", "poor structure"],
        ["positive", "positive", "negative", "negative"],
        seed=42,
    )

    version_dir = tmp_path / "registry_model" / "v1"
    clf.save(version_dir / "model")
    return _make_model_meta(storage_path=version_dir)


def test_run_inference_supports_xlm_roberta_with_stub_backend(tmp_path, monkeypatch) -> None:
    model_meta = _save_stub_xlm_model(tmp_path, monkeypatch)
    df = pd.DataFrame({"text_feedback": ["excellent support", "poor pacing"]})
    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})

    result = run_inference(
        df=df,
        model_meta=model_meta,
        resolved_columns=resolved_columns,
    )

    assert len(result["predictions"]) == 2
    assert len(result["confidences"]) == 2
    assert set(result["classes"]) == {"negative", "positive"}
    assert result["n_predicted"] == 2


def test_compute_or_load_embeddings_uses_xlm_model_embeddings(tmp_path, monkeypatch) -> None:
    model_meta = _save_stub_xlm_model(tmp_path, monkeypatch)
    df = pd.DataFrame(
        {
            "text_feedback": ["excellent support", "poor pacing"],
            "sentiment_class": ["positive", "negative"],
        }
    )

    payload = compute_or_load_embeddings(
        df=df,
        text_col="text_feedback",
        analysis_dir=tmp_path / "analysis",
        model_meta=model_meta,
        reuse_cached=False,
    )

    assert payload["count"] == 2
    assert payload["metadata"]["vector_source"] == f"model_embeddings:{model_meta.id}:mean"
    assert payload["metadata"]["n_features"] == _FakeEncoderConfig.hidden_size
    assert Path(payload["artifact_path"]).exists()
    assert {"row_idx", "x", "y"}.issubset(payload["columns"])


def test_xlm_explainability_returns_clean_not_supported_message(tmp_path) -> None:
    model_meta = _make_model_meta(storage_path=tmp_path)

    with pytest.raises(ValueError, match="not supported"):
        get_global_feature_importance(model_meta)

    with pytest.raises(ValueError, match="not supported"):
        explain_text_instance(model_meta, text="helpful course")
