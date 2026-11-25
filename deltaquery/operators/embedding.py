"""Embedder interface (vendor-neutral) and a deterministic hashing embedder.

`HashingEmbedder` is feature hashing over character n-grams. It is deterministic and needs no
model download, which makes it suitable for tests and as a lexical-ish dense baseline. It is
NOT a semantic model; do not report it as one. A real model (e.g. BGE) plugs in behind the same
`Embedder` protocol and must expose a `model_version` that changes whenever its weights,
tokenizer or pooling change.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Sequence
from typing import Protocol

Vector = tuple[float, ...]


class Embedder(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class EmbeddingError(ValueError):
    pass


def l2_normalize(v: Sequence[float]) -> Vector:
    if any(not math.isfinite(x) for x in v):
        raise EmbeddingError("embedding contains non-finite values")
    n = math.sqrt(math.fsum(x * x for x in v))
    return tuple(0.0 for _ in v) if n == 0 else tuple(x / n for x in v)


def check_vectors(vectors: Sequence[Sequence[float]], *, count: int, dim: int) -> list[Vector]:
    if len(vectors) != count:
        raise EmbeddingError(f"expected {count} vectors, got {len(vectors)}")
    out: list[Vector] = []
    for v in vectors:
        if len(v) != dim:
            raise EmbeddingError(f"expected dim {dim}, got {len(v)}")
        out.append(l2_normalize(v))
    return out


class HashingEmbedder:
    def __init__(self, *, dim: int = 256, ngram: int = 3) -> None:
        if dim < 8 or ngram < 1:
            raise ValueError("bad embedder configuration")
        self._dim = dim
        self.ngram = ngram

    @property
    def model_version(self) -> str:
        return f"hashing-char{self.ngram}-d{self._dim}-v1"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> Vector:
        t = " " + " ".join(unicodedata.normalize("NFC", text).casefold().split()) + " "
        counts = [0] * self._dim
        for i in range(max(0, len(t) - self.ngram + 1)):
            h = int.from_bytes(
                hashlib.blake2b(t[i : i + self.ngram].encode(), digest_size=8).digest(), "big"
            )
            counts[h % self._dim] += 1 if (h >> 63) & 1 else -1
        return l2_normalize([float(c) for c in counts])

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]
