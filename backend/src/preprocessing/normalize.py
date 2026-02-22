"""Unicode and punctuation normalization for multilingual feedback text."""
from __future__ import annotations

import re
import unicodedata


# Typographic quotes → straight quotes
_QUOTE_MAP = str.maketrans({
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u00ab": '"',  # left-pointing double angle quotation mark (Russian «)
    "\u00bb": '"',  # right-pointing double angle quotation mark (Russian »)
    "\u201e": '"',  # double low-9 quotation mark
    "\u201a": "'",  # single low-9 quotation mark
})

# Various dashes → ASCII hyphen-minus
_DASH_RE = re.compile(r"[\u2012\u2013\u2014\u2015\u2212]")

# Ellipsis variants → three dots
_ELLIPSIS_RE = re.compile(r"\u2026|\.{4,}")

# Collapse multiple whitespace characters (including non-breaking space) into a single space
_WHITESPACE_RE = re.compile(r"[\s\u00a0\u200b\u200c\u200d\ufeff]+")


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def normalize_punctuation(text: str) -> str:
    """
    Normalize typographic punctuation to ASCII equivalents:
    - Curly quotes → straight quotes
    - Long dashes → hyphen-minus
    - Ellipsis → three dots
    - Collapse whitespace
    """
    text = text.translate(_QUOTE_MAP)
    text = _DASH_RE.sub("-", text)
    text = _ELLIPSIS_RE.sub("...", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Full text preprocessing pipeline: unicode normalization → punctuation normalization."""
    if not text or not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    return text
