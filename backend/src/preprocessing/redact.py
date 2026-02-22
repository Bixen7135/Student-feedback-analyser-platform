"""Deterministic regex-based PII redaction for student feedback text."""
from __future__ import annotations

import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"

# Email addresses
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Phone numbers: international format, Kazakh/Russian formats
# Matches: +7XXXXXXXXXX, 8XXXXXXXXXX, +7 (XXX) XXX-XX-XX, etc.
_PHONE_RE = re.compile(
    r"(?:\+7|8)[\s\-\(]?\d{3}[\s\-\)]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    r"|"
    r"\b\d{3}[\s\-]\d{2}[\s\-]\d{2}\b",
    re.IGNORECASE,
)

# Patterns that look like student IDs (e.g., "БИ-123456", "ID: 12345")
_STUDENT_ID_RE = re.compile(
    r"\b(?:[А-ЯA-Z]{1,3}[-\s]?\d{5,8}|[Ii][Dd]\s*[:=]?\s*\d{4,10})\b"
)

# URLs — redact to avoid PII in domains
_URL_RE = re.compile(
    r"https?://[^\s]+"
    r"|"
    r"www\.[^\s]+"
)

_PATTERNS: list[re.Pattern] = [_EMAIL_RE, _PHONE_RE, _STUDENT_ID_RE, _URL_RE]


@dataclass
class RedactionResult:
    text: str
    n_redacted: int


def redact_pii(text: str) -> str:
    """Replace PII (emails, phones, student IDs, URLs) with [REDACTED]."""
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_pii_with_count(text: str) -> RedactionResult:
    """Redact PII and return the cleaned text with the count of replacements."""
    n = 0
    for pattern in _PATTERNS:
        matches = pattern.findall(text)
        n += len(matches)
        text = pattern.sub(REDACTED, text)
    return RedactionResult(text=text, n_redacted=n)
