"""Shared fixtures: frozen corpus, BM25 index, operator registry and hand-written plans.

The corpus is a fictional, test-only fixture. Nothing here is an application.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from deltaquery.evidence.corpus import Corpus
from deltaquery.evidence.models import Artifact, Evidence
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext, Operator
from deltaquery.operators.bm25 import BM25Index, BM25Operator, BM25Params
from deltaquery.operators.merge import MergeOperator, MergeParams
from deltaquery.runtime.executor import Executor
from deltaquery.runtime.models import Node, Plan

CORPUS_PATH = Path(__file__).parent / "fixtures" / "corpus.json"
DR_QUERY = "disaster recovery failover RPO RTO"
SCOPE = "demo"


class CountingBM25(BM25Operator):
    def __init__(self, index: BM25Index) -> None:
        super().__init__(index)
        self.calls: list[str] = []

    async def run(
        self, params: BM25Params, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        self.calls.append(ctx.signature)
        return await super().run(params, inputs, ctx)


class CountingMerge(MergeOperator):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(
        self, params: MergeParams, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        self.calls.append(ctx.signature)
        return await super().run(params, inputs, ctx)


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
def index(corpus: Corpus) -> BM25Index:
    return BM25Index(corpus)


@dataclass
class Rig:
    store: EvidenceStore
    bm25: CountingBM25
    merge: CountingMerge
    executor: Executor


def make_rig(index: BM25Index) -> Rig:
    store = EvidenceStore(":memory:")
    bm25 = CountingBM25(index)
    merge = CountingMerge()
    ops: dict[str, Operator[Any]] = {"bm25": bm25, "merge": merge}
    return Rig(store, bm25, merge, Executor(store, ops, clock=TickClock()))


@pytest.fixture
def rig(index: BM25Index) -> Rig:
    return make_rig(index)


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
