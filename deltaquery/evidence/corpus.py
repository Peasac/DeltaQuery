"""Frozen source corpus with stable source ids, version hashes and a snapshot id."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from deltaquery.canonical import hash_canonical, normalize_entity_id, sha256_hex
from deltaquery.evidence.models import Span

CHUNKER_VERSION = "paragraph-v1"

# A paragraph: a run of non-blank lines.
_PARAGRAPH = re.compile(r"[^\n]*\S[^\n]*(?:\n[^\n]*\S[^\n]*)*")


class SourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    entity: str
    title: str
    text: str
    access_scope: str

    @field_validator("entity")
    @classmethod
    def _norm_entity(cls, v: str) -> str:
        return normalize_entity_id(v)

    @property
    def source_version(self) -> str:
        return hash_canonical("source", self.model_dump(mode="python"))


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    source_version: str
    entity: str
    access_scope: str
    span: Span
    text: str

    @property
    def content_hash(self) -> str:
        return sha256_hex(self.text)


def chunk_paragraphs(doc: SourceDocument) -> list[Passage]:
    out: list[Passage] = []
    version = doc.source_version
    for m in _PARAGRAPH.finditer(doc.text):
        raw = m.group(0)
        lead = len(raw) - len(raw.lstrip())
        text = raw.strip()
        start = m.start() + lead
        out.append(
            Passage(
                source_id=doc.source_id,
                source_version=version,
                entity=doc.entity,
                access_scope=doc.access_scope,
                span=Span(start=start, end=start + len(text)),
                text=text,
            )
        )
    return out


class Corpus:
    """Immutable set of documents. `snapshot_id` changes iff any document changes."""

    def __init__(self, documents: Iterable[SourceDocument]) -> None:
        docs = sorted(documents, key=lambda d: d.source_id)
        ids = [d.source_id for d in docs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source_id in corpus")
        self._docs: tuple[SourceDocument, ...] = tuple(docs)

    @property
    def documents(self) -> tuple[SourceDocument, ...]:
        return self._docs

    @cached_property
    def snapshot_id(self) -> str:
        return hash_canonical("snapshot", [[d.source_id, d.source_version] for d in self._docs])

    def passages(self) -> list[Passage]:
        return [p for d in self._docs for p in chunk_paragraphs(d)]

    @classmethod
    def from_json(cls, path: Path) -> Corpus:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(SourceDocument.model_validate(item) for item in raw)


class PassageSet:
    """The shared span representation: every retriever over a corpus indexes exactly these
    passages, so evidence for the same text has the same (source_id, source_version, span,
    content_hash) regardless of which retriever found it."""

    def __init__(self, corpus: Corpus) -> None:
        self.source_snapshot = corpus.snapshot_id
        self.passages: tuple[Passage, ...] = tuple(corpus.passages())
        self._domains: dict[tuple[str | None, str], frozenset[str]] = {}
        self.version = hash_canonical(
            "passages", {"snapshot": self.source_snapshot, "chunker": CHUNKER_VERSION}
        )

    def domain_sources(self, *, entity: str | None, access_scope: str) -> frozenset[str]:
        """Source ids a retriever filtered by (entity, scope) could ever return."""
        key = (entity, access_scope)
        cached = self._domains.get(key)
        if cached is None:
            cached = frozenset(
                p.source_id
                for p in self.passages
                if p.access_scope == access_scope and (entity is None or p.entity == entity)
            )
            self._domains[key] = cached
        return cached

    def candidates(
        self,
        *,
        entity: str | None,
        access_scope: str,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> Iterator[tuple[int, Passage]]:
        """Passages visible in `access_scope` (exact match, fail closed), minus revoked
        sources, optionally restricted to one entity."""
        for i, p in enumerate(self.passages):
            if p.access_scope != access_scope or p.source_id in exclude_sources:
                continue
            if entity is not None and p.entity != entity:
                continue
            yield i, p
