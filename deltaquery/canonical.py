"""Canonical encoding and content-addressed hashing.

Everything that is hashed goes through `canonical_json`: keys sorted, strings NFC-normalized,
floats encoded by their shortest round-trip repr (with -0.0 folded to 0.0), and non-finite
floats rejected. Hashes are domain-separated by a `kind` tag so that, e.g., a node signature
can never collide with an artifact hash over the same payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import BaseModel

Canonical: TypeAlias = bool | int | str | list["Canonical"] | dict[str, "Canonical"] | None

_FLOAT_TAG = "$float"


class CanonicalizationError(ValueError):
    """A value cannot be encoded canonically (and therefore must not be hashed)."""


def canonicalize(value: object) -> Canonical:
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python"))
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite float: {value!r}")
        return {_FLOAT_TAG: repr(0.0 if value == 0 else value)}
    if isinstance(value, Mapping):
        out: dict[str, Canonical] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string key: {key!r}")
            if key.startswith("$"):
                raise CanonicalizationError(f"reserved key prefix '$': {key!r}")
            norm_key = unicodedata.normalize("NFC", key)
            if norm_key in out:
                raise CanonicalizationError(f"keys collide after normalization: {key!r}")
            out[norm_key] = canonicalize(item)
        return out
    if isinstance(value, list | tuple):
        return [canonicalize(item) for item in value]
    raise CanonicalizationError(f"unsupported type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(
        canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_canonical(kind: str, value: object) -> str:
    """Domain-separated content hash of a canonically encoded value."""
    return sha256_hex(f"deltaquery/{kind}/v1\n{canonical_json(value)}")


_WS = re.compile(r"\s+")


def normalize_entity_id(raw: str) -> str:
    """Normalize an entity identifier: NFC, casefold, trim, whitespace -> '-'."""
    norm = _WS.sub("-", unicodedata.normalize("NFC", raw).casefold().strip())
    if not norm:
        raise ValueError("empty entity id")
    return norm
