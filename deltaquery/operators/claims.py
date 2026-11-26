"""Deterministic typed claim extraction (no LLM).

A claim is a typed, exact value stated in a passage, e.g. `rpo=30s`. Extraction is deliberately
conservative: hedged or approximate statements ("under one minute", "about thirty seconds",
"typically ...") yield *no* claim, because a coverage decision built on them could skip
retrieval that was actually needed. No claim means "not covered", which means retrieve.

Bump EXTRACTOR_VERSION whenever patterns change: it is part of every signature that uses it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from deltaquery.domain import DEFAULT_DOMAIN

EXTRACTOR_VERSION = "regex-v1"

_NUMBER_WORDS = {"zero": 0}
_UNIT_SECONDS = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
}
_HEDGES = re.compile(
    r"\b(under|about|approximately|approx|around|roughly|less than|up to|typically|usually|"
    r"near|nearly|almost|over|more than)\b",
    re.I,
)
_LABELS = DEFAULT_DOMAIN.claim_labels
_VALUE = r"(\d+|zero)\s*(seconds?|minutes?|hours?)\b"

ATTRIBUTES = tuple(_LABELS)


def compile_patterns(labels: Mapping[str, str]) -> dict[str, re.Pattern[str]]:
    # label, then a short gap inside the same sentence, then the value
    return {a: re.compile(rf"{lab}([^.\n]{{0,60}}?)\b{_VALUE}", re.I) for a, lab in labels.items()}


_PATTERNS = compile_patterns(_LABELS)


def extract_claims(
    text: str,
    attributes: tuple[str, ...] = ATTRIBUTES,
    patterns: Mapping[str, re.Pattern[str]] | None = None,
) -> tuple[str, ...]:
    """Exact claims like ('rpo=30s',) found in `text` for the given attributes. `patterns`
    defaults to the default domain's labels; an attribute without a pattern raises KeyError."""
    pats = _PATTERNS if patterns is None else patterns
    out: list[str] = []
    for attr in attributes:
        m = pats[attr].search(text)
        if m is None:
            continue
        gap, number, unit = m.group(1), m.group(2).lower(), m.group(3).lower()
        if _HEDGES.search(gap):
            continue
        value = _NUMBER_WORDS[number] if number in _NUMBER_WORDS else int(number)
        out.append(f"{attr}={value * _UNIT_SECONDS[unit]}s")
    return tuple(out)
