"""Dense retrieval over the shared passage set: exact cosine search (no ANN in the MVP).

At ~tens of passages exact search is cheap and, unlike an ANN index, has no approximation that
could differ between full and incremental runs. FAISS/Qdrant can replace `search` later behind
the same operator, with its own index version.

Query canonicalization is deliberately weak (NFC + whitespace collapse only): an arbitrary
embedding model may be case- or punctuation-sensitive, so anything stronger could merge queries
that are not the same computation.
"""

from __future__ import annotations

import asyncio
import math
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deltaquery.canonical import hash_canonical, normalize_entity_id
from deltaquery.evidence.corpus import Passage, PassageSet
from deltaquery.evidence.models import Artifact, CoverageType, Evidence
from deltaquery.operators.base import (
    RANKED_PASSAGES,
    BoundVersions,
    ExecutionContext,
    Operator,
    passage_evidence,
    rank,
)
from deltaquery.operators.embedding import Embedder, Vector, check_vectors

SEARCH_VERSION = "exact-cosine-v1"


def canonical_dense_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class DenseParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    top_k: int = Field(ge=1, le=100)
    entity: str | None = None

    @field_validator("query")
    @classmethod
    def _canon_query(cls, v: str) -> str:
        q = canonical_dense_query(v)
        if not q:
            raise ValueError("empty query")
        return q

    @field_validator("entity")
    @classmethod
    def _canon_entity(cls, v: str | None) -> str | None:
        return None if v is None else normalize_entity_id(v)


def _dot(a: Vector, b: Vector) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


class DenseIndex:
    """Use `await DenseIndex.build(...)`; passage vectors are computed once per snapshot."""

    def __init__(self, passages: PassageSet, embedder: Embedder, vectors: Sequence[Vector]):
        if len(vectors) != len(passages.passages):
            raise ValueError("one vector per passage required")
        self.passages = passages
        self.embedder = embedder
        self.snapshot_id = passages.source_snapshot
        self.model_version = embedder.model_version
        self._vectors = tuple(vectors)
        self.version = hash_canonical(
            "index",
            {
                "kind": "dense",
                "passages": passages.version,
                "model": embedder.model_version,
                "search": SEARCH_VERSION,
            },
        )

    @classmethod
    async def build(
        cls, passages: PassageSet, embedder: Embedder, *, timeout_s: float = 120.0
    ) -> DenseIndex:
        texts = [p.text for p in passages.passages]
        raw = await asyncio.wait_for(embedder.embed(texts), timeout=timeout_s)
        return cls(passages, embedder, check_vectors(raw, count=len(texts), dim=embedder.dim))

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        entity: str | None,
        access_scope: str,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> list[tuple[Passage, float]]:
        raw = await self.embedder.embed([query])
        (q,) = check_vectors(raw, count=1, dim=self.embedder.dim)
        scored = [
            (p, _dot(q, self._vectors[i]))
            for i, p in self.passages.candidates(
                entity=entity, access_scope=access_scope, exclude_sources=exclude_sources
            )
        ]
        return rank(scored, top_k)


class DenseOperator(Operator[DenseParams]):
    name = "dense"
    version = "1"
    output_contract = RANKED_PASSAGES
    params_model = DenseParams
    min_inputs = 0
    max_inputs = 0

    def __init__(self, index: DenseIndex) -> None:
        self.index = index

    @property
    def bound_versions(self) -> BoundVersions:
        return BoundVersions(
            source_snapshot=self.index.snapshot_id,
            index_version=self.index.version,
            model_version=self.index.model_version,
        )

    def access_dependencies(
        self, params: DenseParams, access_scope: str, revoked: frozenset[str]
    ) -> frozenset[str]:
        # A passage's score does not depend on which other passages are excluded
        # (cosine against a fixed query vector), so only revocations inside the filtered
        # domain can change the top-k.
        domain = self.index.passages.domain_sources(entity=params.entity, access_scope=access_scope)
        return revoked & domain

    async def run(
        self, params: DenseParams, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        hits = await self.index.search(
            params.query,
            top_k=params.top_k,
            entity=params.entity,
            access_scope=ctx.access_scope,
            exclude_sources=ctx.excluded_sources,
        )
        return [passage_evidence(p, score=s, coverage=CoverageType.DENSE, ctx=ctx) for p, s in hits]
