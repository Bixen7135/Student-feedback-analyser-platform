"""Unit tests for text classification tasks."""
from __future__ import annotations

import numpy as np
import pytest


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
