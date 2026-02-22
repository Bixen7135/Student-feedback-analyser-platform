"""Unit tests for preprocessing modules."""
from __future__ import annotations

import pytest
from src.preprocessing.normalize import normalize_unicode, normalize_punctuation, preprocess_text
from src.preprocessing.redact import redact_pii
from src.preprocessing.features import compute_text_features


def test_normalize_unicode_nfc():
    text = "cafe\u0301"  # NFD: e + combining accent
    result = normalize_unicode(text)
    assert result == "caf\u00e9"  # NFC: single character


def test_normalize_unicode_already_nfc():
    text = "Привет мир"
    assert normalize_unicode(text) == text


def test_normalize_punctuation_dashes():
    text = "это \u2013 тире"
    result = normalize_punctuation(text)
    assert "-" in result
    assert "\u2013" not in result


def test_normalize_punctuation_ellipsis():
    text = "и так далее\u2026"
    result = normalize_punctuation(text)
    assert "..." in result


def test_normalize_punctuation_curly_quotes():
    text = "\u00abhорошо\u00bb"  # Russian «хорошо»
    result = normalize_punctuation(text)
    assert '"' in result
    assert "\u00ab" not in result


def test_preprocess_text_full_pipeline():
    text = "  Хорошо\u2026  test@email.com  "
    result = preprocess_text(text)
    assert "test@email.com" in result  # redaction happens separately
    assert "\u2026" not in result
    assert not result.startswith(" ")


def test_redact_email():
    text = "Напишите нам на student@university.kz"
    result = redact_pii(text)
    assert "student@university.kz" not in result
    assert "[REDACTED]" in result


def test_redact_phone_number():
    text = "Позвоните +7 701 123 45 67"
    result = redact_pii(text)
    assert "701 123 45 67" not in result
    assert "[REDACTED]" in result


def test_redact_preserves_normal_text():
    text = "Отличный курс, спасибо преподавателям!"
    result = redact_pii(text)
    assert result == text


def test_compute_text_features_word_count():
    text = "Хороший курс но можно лучше"
    features = compute_text_features(text)
    assert features.word_count == 5
    assert features.char_count == len(text)


def test_compute_text_features_empty():
    features = compute_text_features("")
    assert features.word_count == 0
    assert features.char_count == 0
    assert features.avg_word_length == 0.0
