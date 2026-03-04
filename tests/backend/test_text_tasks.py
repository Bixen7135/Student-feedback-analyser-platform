"""Unit tests for text classification tasks."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest


class _FakeTokenizer:
    def __init__(self, source: str) -> None:
        self.source = source

    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.txt").write_text(self.source, encoding="utf-8")


class _FakeEncoderConfig:
    hidden_size = 8


class _FakeEncoder:
    def __init__(self, source: str) -> None:
        self.source = source
        self.config = _FakeEncoderConfig()

    def save_pretrained(self, path) -> None:
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
        rows = [
            network.class_bias + (len(text) * scale)
            for text in texts
        ]
        return np.vstack(rows)

    def save_pretrained(self, *, network: _FakeNetwork, tokenizer: _FakeTokenizer, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        network.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "tokenizer")
        (path / "head.pt").write_text(
            json.dumps(network.state_dict()),
            encoding="utf-8",
        )

    def load_weights(self, *, network: _FakeNetwork, path) -> None:
        payload = json.loads((path / "head.pt").read_text(encoding="utf-8"))
        network.load_state_dict(payload)

    def copy_state(self, *, source: _FakeNetwork, target: _FakeNetwork) -> None:
        target.load_state_dict(source.state_dict())


class _FakeTransformersLogging:
    def __init__(self) -> None:
        self.verbosity = 20
        self.progress_disabled = False
        self.events: list[str] = []

    def get_verbosity(self) -> int:
        self.events.append("get_verbosity")
        return self.verbosity

    def set_verbosity(self, value: int) -> None:
        self.events.append(f"set_verbosity:{value}")
        self.verbosity = value

    def set_verbosity_error(self) -> None:
        self.events.append("set_verbosity_error")
        self.verbosity = 40

    def disable_progress_bar(self) -> None:
        self.events.append("disable_progress_bar")
        self.progress_disabled = True

    def enable_progress_bar(self) -> None:
        self.events.append("enable_progress_bar")
        self.progress_disabled = False


def _transformer_runtime_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def test_tfidf_classifier_fit_predict(preprocessed_df):
    from src.text_tasks.tfidf_classifier import TfidfClassifier
    clf = TfidfClassifier(max_features=100)
    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["sentiment_class"].tolist()
    clf.fit(texts, labels, seed=42)
    preds = clf.predict(texts)
    assert len(preds) == len(texts)
    assert set(preds).issubset({"positive", "neutral", "negative"})


def test_char_ngram_classifier_fit_predict(preprocessed_df):
    from src.text_tasks.char_ngram_classifier import CharNgramClassifier
    clf = CharNgramClassifier(max_features=500)
    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["language"].tolist()
    clf.fit(texts, labels, seed=42)
    preds = clf.predict(texts)
    assert len(preds) == len(texts)
    assert set(preds).issubset({"ru", "kz", "mixed"})


def test_tfidf_deterministic_with_seed(preprocessed_df):
    from src.text_tasks.tfidf_classifier import TfidfClassifier
    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["sentiment_class"].tolist()

    clf1 = TfidfClassifier(max_features=100)
    clf1.fit(texts, labels, seed=42)
    preds1 = clf1.predict(texts)

    clf2 = TfidfClassifier(max_features=100)
    clf2.fit(texts, labels, seed=42)
    preds2 = clf2.predict(texts)

    assert list(preds1) == list(preds2)


def test_classifier_save_load_roundtrip(preprocessed_df, tmp_path):
    from src.text_tasks.tfidf_classifier import TfidfClassifier
    clf = TfidfClassifier(max_features=100)
    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["sentiment_class"].tolist()
    clf.fit(texts, labels, seed=42)
    preds_before = clf.predict(texts)

    model_path = tmp_path / "model.joblib"
    clf.save(model_path)
    clf2 = TfidfClassifier.load(model_path)
    preds_after = clf2.predict(texts)
    assert list(preds_before) == list(preds_after)


def test_predict_proba_shape(preprocessed_df):
    from src.text_tasks.tfidf_classifier import TfidfClassifier
    clf = TfidfClassifier(max_features=100)
    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["sentiment_class"].tolist()
    clf.fit(texts, labels, seed=42)
    proba = clf.predict_proba(texts)
    assert proba.shape[0] == len(texts)
    assert proba.shape[1] == 3  # positive, neutral, negative
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_xlm_roberta_classifier_dependency_error_is_clear(monkeypatch):
    from src.text_tasks.xlm_roberta_classifier import (
        XlmRobertaClassifier,
        XlmRobertaDependencyError,
    )

    monkeypatch.setattr(
        XlmRobertaClassifier,
        "_backend_factory",
        staticmethod(
            lambda: (_ for _ in ()).throw(
                XlmRobertaDependencyError(
                    "XlmRobertaClassifier requires optional dependencies 'torch' and "
                    "'transformers'. Install with: pip install \"sfap-backend[transformers]\""
                )
            )
        ),
    )

    clf = XlmRobertaClassifier()
    with pytest.raises(XlmRobertaDependencyError, match="sfap-backend\\[transformers\\]"):
        clf.fit(
            ["good course", "bad course"],
            ["positive", "negative"],
            seed=42,
        )


def test_xlm_roberta_classifier_roundtrip_with_stub_backend(preprocessed_df, tmp_path, monkeypatch):
    from src.text_tasks.xlm_roberta_classifier import XlmRobertaClassifier

    monkeypatch.setattr(
        XlmRobertaClassifier,
        "_backend_factory",
        staticmethod(lambda: _FakeXlmBackend()),
    )

    texts = preprocessed_df["text_processed"].fillna("").tolist()
    labels = preprocessed_df["sentiment_class"].tolist()
    clf = XlmRobertaClassifier(
        pretrained_model="xlm-roberta-base",
        max_seq_length=64,
        batch_size=4,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        head_hidden_units=16,
        dropout=0.2,
        activation="relu",
    )
    clf.fit(texts, labels, seed=42)
    preds_before = clf.predict(texts)
    proba_before = clf.predict_proba(texts)

    model_path = tmp_path / "xlm_model"
    clf.save(model_path)

    loaded = XlmRobertaClassifier.load(model_path)
    preds_after = loaded.predict(texts)
    proba_after = loaded.predict_proba(texts)

    assert loaded.classes_ == clf.classes_
    assert list(preds_before) == list(preds_after)
    assert proba_before.shape == proba_after.shape
    assert np.allclose(proba_before, proba_after)


def test_xlm_roberta_runtime_quiets_transformer_load_reports():
    from types import SimpleNamespace

    from src.text_tasks.xlm_roberta_classifier import _TorchRuntime

    logging = _FakeTransformersLogging()
    calls: list[str] = []

    class _Loader:
        @staticmethod
        def from_pretrained(source: str) -> str:
            calls.append(source)
            assert logging.verbosity == 40
            assert logging.progress_disabled is True
            return f"loaded:{source}"

    runtime = _TorchRuntime(
        torch=None,
        nn=None,
        transformers=SimpleNamespace(utils=SimpleNamespace(logging=logging)),
        AutoModel=_Loader,
        AutoTokenizer=_Loader,
        AdamW=None,
    )

    tokenizer = runtime.create_tokenizer("tokenizer-src")
    encoder = runtime.create_encoder("encoder-src")

    assert tokenizer == "loaded:tokenizer-src"
    assert encoder == "loaded:encoder-src"
    assert calls == ["tokenizer-src", "encoder-src"]
    assert logging.verbosity == 20
    assert logging.progress_disabled is False
    assert logging.events == [
        "get_verbosity",
        "disable_progress_bar",
        "set_verbosity_error",
        "set_verbosity:20",
        "enable_progress_bar",
        "get_verbosity",
        "disable_progress_bar",
        "set_verbosity_error",
        "set_verbosity:20",
        "enable_progress_bar",
    ]


@pytest.mark.skipif(
    not _transformer_runtime_available(),
    reason="Optional transformer dependencies are not installed. Run `uv sync --extra transformers`.",
)
def test_xlm_roberta_runtime_imports_when_transformer_extra_is_installed():
    from src.text_tasks.xlm_roberta_classifier import _TorchRuntime

    runtime = _TorchRuntime.load()

    assert runtime.torch is not None
    assert runtime.AutoModel is not None
    assert runtime.AutoTokenizer is not None


def test_train_all_baselines_returns_all_tasks(preprocessed_df, tmp_path):
    from src.splits.splitter import stratified_split
    from src.text_tasks.trainer import train_all_baselines
    train_df, val_df, _ = stratified_split(preprocessed_df, seed=42)
    results = train_all_baselines(train_df, val_df, tmp_path, seed=42)
    assert "language" in results
    assert "sentiment" in results
    assert "detail_level" in results
    for task, task_results in results.items():
        assert "tfidf" in task_results
        assert "char_ngram" in task_results
