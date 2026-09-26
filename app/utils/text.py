"""Shared text normalization helpers for grounding checks."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase for containment checks."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def text_supported(needle: str, haystack: str) -> bool:
    """Return True when needle is supported by haystack after light normalization.

    Exact normalized containment wins; otherwise require most content tokens
    (length > 2) to appear so minor wrapping differences are tolerated.
    """
    normalized_needle = normalize_text(needle)
    normalized_haystack = normalize_text(haystack)
    if not normalized_needle:
        return False
    if normalized_needle in normalized_haystack:
        return True

    tokens = [token for token in normalized_needle.split() if len(token) > 2]
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in normalized_haystack)
    return matched / len(tokens) >= 0.8
