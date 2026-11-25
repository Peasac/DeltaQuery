"""Shared fixtures: frozen corpus, BM25 index, operator registry and hand-written plans.

The corpus is a fictional, test-only fixture. Nothing here is an application.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from deltaquery.evidence.corpus import Corpus, PassageSet
from deltaquery.evidence.models import Artifact
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext, Operator
from deltaquery.operators.bm25 import BM25Index, BM25Operator, BM25Params
from deltaquery.operators.dense import DenseIndex, DenseOperator, DenseParams
from deltaquery.operators.embedding import HashingEmbedder
from deltaquery.operators.fusion import RRFOperator, RRFParams
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.executor import Executor
from deltaquery.runtime.models import Node, Plan

CORPUS_PATH = Path(__file__).parent / "fixtures" / "corpus.json"
DR_QUERY = "disaster recovery failover RPO RTO"
SCOPE = "demo"


class Counted:
    """Mixin recording the signature of every node an operator actually executes."""

    calls: list[str]

    async def run(self, params: Any, inputs: Sequence[Artifact], ctx: ExecutionContext) -> Any:
        self.calls.append(ctx.signature)
        return await super().run(params, inputs, ctx)  # type: ignore[misc]


class CountingBM25(Counted, BM25Operator):
    def __init__(self, index: BM25Index) -> None:
        super().__init__(index)
        self.calls = []


class CountingDense(Counted, DenseOperator):
    def __init__(self, index: DenseIndex) -> None:
        super().__init__(index)
        self.calls = []


class CountingRRF(Counted, RRFOperator):
    def __init__(self) -> None:
        self.calls = []


class CountingMerge(Counted, MergeOperator):
    def __init__(self) -> None:
        self.calls = []


@dataclass
class TickClock:
    """Deterministic clock that advances one second per call (proves created_at is not identity)."""

    t: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=1)
        return self.t


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.from_json(CORPUS_PATH)


@pytest.fixture(scope="session")
def passages(corpus: Corpus) -> PassageSet:
    return PassageSet(corpus)


@pytest.fixture(scope="session")
def index(passages: PassageSet) -> BM25Index:
    return BM25Index(passages)


@pytest.fixture(scope="session")
def dense_index(passages: PassageSet) -> DenseIndex:
    return asyncio.run(DenseIndex.build(passages, HashingEmbedder()))


@dataclass
class Rig:
    store: EvidenceStore
    bm25: CountingBM25
    dense: CountingDense | None
    rrf: CountingRRF
    merge: CountingMerge
    executor: Executor


def make_rig(index: BM25Index, dense_index: DenseIndex | None = None, *extra: Operator[Any]) -> Rig:
    store = EvidenceStore(":memory:")
    bm25 = CountingBM25(index)
    dense = CountingDense(dense_index) if dense_index is not None else None
    rrf, merge = CountingRRF(), CountingMerge()
    ops: list[Operator[Any]] = [bm25, rrf, merge, *extra]
    if dense is not None:
        ops.append(dense)
    return Rig(store, bm25, dense, rrf, merge, Executor(store, ops, clock=TickClock()))


@pytest.fixture
def rig(index: BM25Index, dense_index: DenseIndex) -> Rig:
    return make_rig(index, dense_index)


# -- hand-written plans -------------------------------------------------------------------


def retrieve(
    node_id: str,
    index: BM25Index,
    entity: str,
    *,
    query: str = DR_QUERY,
    top_k: int = 3,
    scope: str = SCOPE,
) -> Node:
    params = BM25Params(query=query, top_k=top_k, entity=entity)
    return Node(
        id=node_id,
        operator="bm25",
        canonical_parameters=params.model_dump(mode="json"),
        source_snapshot=index.snapshot_id,
        index_version=index.version,
        model_version=None,
        access_scope=scope,
        output_contract="ranked_passages",
    )


def merge(node_id: str, *inputs: str, scope: str = SCOPE) -> Node:
    return Node(
        id=node_id,
        operator="merge",
        canonical_parameters={},
        input_ids=tuple(inputs),
        source_snapshot=None,
        index_version=None,
        model_version=None,
        access_scope=scope,
        output_contract="evidence_set",
    )


def plan_compare(index: BM25Index, left: str, right: str, **kw: Any) -> Plan:
    """Q1-style plan: A=retrieve(left), B/C=retrieve(right), D=merge(A, B/C)."""
    right_id = {"azure": "B", "gcp": "C"}.get(right.casefold(), "R")
    return Plan(
        nodes=(
            retrieve("A", index, left, **kw),
            retrieve(right_id, index, right, **kw),
            merge("D", "A", right_id, scope=kw.get("scope", SCOPE)),
        ),
        output_id="D",
    )


def dense(
    node_id: str,
    index: DenseIndex,
    entity: str,
    *,
    query: str = DR_QUERY,
    top_k: int = 3,
    scope: str = SCOPE,
) -> Node:
    params = DenseParams(query=query, top_k=top_k, entity=entity)
    return Node(
        id=node_id,
        operator="dense",
        canonical_parameters=params.model_dump(mode="json"),
        source_snapshot=index.snapshot_id,
        index_version=index.version,
        model_version=index.model_version,
        access_scope=scope,
        output_contract="ranked_passages",
    )


def rrf(node_id: str, *inputs: str, k: int = 60, top_k: int = 3, scope: str = SCOPE) -> Node:
    return Node(
        id=node_id,
        operator="rrf",
        canonical_parameters=RRFParams(k=k, top_k=top_k).model_dump(mode="json"),
        input_ids=tuple(inputs),
        source_snapshot=None,
        index_version=None,
        model_version=None,
        access_scope=scope,
        output_contract="ranked_passages",
    )


def plan_hybrid(
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    left: str,
    right: str,
    *,
    k: int = 60,
    query: str = DR_QUERY,
) -> Plan:
    """Per entity: lexical + dense candidates -> RRF; then D = merge(left, right)."""
    right_id = {"azure": "B", "gcp": "C"}.get(right.casefold(), "R")
    nodes: list[Node] = []
    for nid, entity in (("A", left), (right_id, right)):
        nodes += [
            retrieve(f"{nid}_lex", bm25_index, entity, query=query, top_k=5),
            dense(f"{nid}_dense", dense_index, entity, query=query, top_k=5),
            rrf(nid, f"{nid}_lex", f"{nid}_dense", k=k),
        ]
    nodes.append(merge("D", "A", right_id))
    return Plan(nodes=tuple(nodes), output_id="D")
