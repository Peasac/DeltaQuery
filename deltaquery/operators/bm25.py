"""Minimal deterministic BM25 over paragraph passages of a frozen corpus.

Determinism: query terms are deduplicated and sorted before scoring (so float summation order
is fixed), and ties are broken by (source_id, span start). Because scoring only depends on the
*set* of query tokens, the canonical query is that sorted token set; two queries that tokenize
to the same set therefore share a signature, and that equivalence is exact, not approximate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deltaquery.canonical import hash_canonical, normalize_entity_id
from deltaquery.evidence.corpus import CHUNKER_VERSION, Corpus, Passage
from deltaquery.evidence.models import Artifact, CoverageType, Evidence
from deltaquery.operators.base import BoundVersions, ExecutionContext, Operator

TOKENIZER_VERSION = "casefold-alnum-v1"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def canonical_query(text: str) -> str:
    return " ".join(sorted(set(tokenize(text))))


class BM25Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    top_k: int = Field(ge=1, le=100)
    entity: str | None = None

    @field_validator("query")
    @classmethod
    def _canon_query(cls, v: str) -> str:
        q = canonical_query(v)
        if not q:
            raise ValueError("query has no searchable tokens")
        return q

    @field_validator("entity")
    @classmethod
    def _canon_entity(cls, v: str | None) -> str | None:
        return None if v is None else normalize_entity_id(v)


class BM25Index:
    def __init__(self, corpus: Corpus, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.snapshot_id = corpus.snapshot_id
        self.passages: tuple[Passage, ...] = tuple(corpus.passages())
        self._tf = [Counter(tokenize(p.text)) for p in self.passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
        self.version = hash_canonical(
            "index",
            {
                "kind": "bm25",
                "snapshot": self.snapshot_id,
                "tokenizer": TOKENIZER_VERSION,
                "chunker": CHUNKER_VERSION,
                "k1": k1,
                "b": b,
            },
        )

    def search(
        self, query: str, *, top_k: int, entity: str | None, access_scope: str
    ) -> list[tuple[Passage, float]]:
        terms = sorted(set(tokenize(query)))
        scored: list[tuple[Passage, float]] = []
        for i, p in enumerate(self.passages):
            if p.access_scope != access_scope:  # fail closed: exact scope match only
                continue
            if entity is not None and p.entity != entity:
                continue
            tf = self._tf[i]
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avgdl)
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((p, score))
        scored.sort(key=lambda ps: (-ps[1], ps[0].source_id, ps[0].span.start))
        return scored[:top_k]


class BM25Operator(Operator[BM25Params]):
    name = "bm25"
    version = "1"
    output_contract = "ranked_passages"
    params_model = BM25Params
    min_inputs = 0
    max_inputs = 0

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    @property
    def bound_versions(self) -> BoundVersions:
        return BoundVersions(
            source_snapshot=self.index.snapshot_id,
            index_version=self.index.version,
            model_version=None,
        )

    async def run(
        self, params: BM25Params, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        hits = self.index.search(
            params.query, top_k=params.top_k, entity=params.entity, access_scope=ctx.access_scope
        )
        return [
            Evidence(
                source_id=p.source_id,
                source_version=p.source_version,
                span=p.span,
                content_hash=p.content_hash,
                retrieval_node_id=ctx.signature,
                score=score,
                coverage_type=CoverageType.LEXICAL,
                created_at=ctx.now,
            )
            for p, score in hits
        ]
