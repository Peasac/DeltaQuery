"""Phase 1 acceptance tests (project.md §12, CLAUDE.md Testing)."""

from __future__ import annotations

import asyncio
from typing import Any

from conftest import PRINCIPAL, Rig, make_rig, plan_compare, retrieve

from deltaquery.evidence.models import Artifact
from deltaquery.operators.bm25 import BM25Index
from deltaquery.runtime.executor import ExecutionResult, NodeStatus
from deltaquery.runtime.models import Plan


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def content(a: Artifact) -> tuple[Any, ...]:
    """Everything about an artifact except bookkeeping timestamps."""
    return (
        a.signature,
        a.artifact_hash,
        a.operator,
        a.operator_version,
        a.input_signatures,
        a.access_scope,
        a.source_snapshot,
        a.index_version,
        a.model_version,
        tuple(e.identity() for e in a.evidence),
    )


def test_identical_query_reuses_all_artifacts(rig: Rig, index: BM25Index) -> None:
    q1 = plan_compare(index, "AWS", "Azure")

    async def scenario() -> tuple[ExecutionResult, ExecutionResult, int, int]:
        first = await rig.executor.execute(q1, principal=PRINCIPAL)
        n1 = await rig.store.count_artifacts()
        second = await rig.executor.execute(q1, principal=PRINCIPAL)
        n2 = await rig.store.count_artifacts()
        return first, second, n1, n2

    first, second, n1, n2 = run(scenario())
    assert first.executed == {"A", "B", "D"} and first.reused == set()
    assert second.reused == {"A", "B", "D"} and second.executed == set()
    assert len(rig.bm25.calls) == 2 and len(rig.merge.calls) == 1  # nothing re-ran
    assert n1 == n2 == 3
    for nid in ("A", "B", "D"):
        assert content(first.artifacts[nid]) == content(second.artifacts[nid])
    assert first.output.evidence, "fixture should actually retrieve something"


def test_equivalent_phrasing_reuses_exactly(rig: Rig, index: BM25Index) -> None:
    # Same token set -> provably identical BM25 computation -> same signature.
    a = Plan(
        nodes=(retrieve("A", index, "AWS", query="RTO rpo, failover: Disaster recovery"),),
        output_id="A",
    )
    b = Plan(
        nodes=(retrieve("A", index, " aws ", query="disaster recovery failover RPO RTO"),),
        output_id="A",
    )

    async def scenario() -> tuple[ExecutionResult, ExecutionResult]:
        return await rig.executor.execute(a, principal=PRINCIPAL), await rig.executor.execute(
            b, principal=PRINCIPAL
        )

    ra, rb = run(scenario())
    assert ra.executed == {"A"} and rb.reused == {"A"}


def test_unrelated_query_executes(rig: Rig, index: BM25Index) -> None:
    q1 = plan_compare(index, "AWS", "Azure")
    other = plan_compare(index, "AWS", "Azure", query="encryption keys at rest")

    async def scenario() -> ExecutionResult:
        await rig.executor.execute(q1, principal=PRINCIPAL)
        return await rig.executor.execute(other, principal=PRINCIPAL)

    r = run(scenario())
    assert r.reused == set()
    assert r.executed == {"A", "B", "D"}


def test_changed_entity_invalidates_only_dependent_nodes(rig: Rig, index: BM25Index) -> None:
    q1 = plan_compare(index, "AWS", "Azure")  # A=aws, B=azure, D=merge(A,B)
    q2 = plan_compare(index, "AWS", "GCP")  # A=aws, C=gcp,   D=merge(A,C)

    async def scenario() -> tuple[ExecutionResult, ExecutionResult, Artifact | None]:
        r1 = await rig.executor.execute(q1, principal=PRINCIPAL)
        r2 = await rig.executor.execute(q2, principal=PRINCIPAL)
        old_d = await rig.store.get_artifact(r1.artifacts["D"].signature)
        return r1, r2, old_d

    r1, r2, old_d = run(scenario())
    # A is independent of the replaced entity: reused, same artifact.
    assert r2.reused == {"A"}
    assert content(r2.artifacts["A"]) == content(r1.artifacts["A"])
    # The new branch and its dependent join are executed.
    assert r2.executed == {"C", "D"}
    assert r2.artifacts["D"].signature != r1.artifacts["D"].signature
    assert r2.artifacts["D"].input_signatures == (
        r2.artifacts["A"].signature,
        r2.artifacts["C"].signature,
    )
    # The removed branch B contributes nothing to the new output.
    assert "B" not in r2.artifacts
    b_sources = {e.source_id for e in r1.artifacts["B"].evidence}
    assert b_sources and not b_sources & {e.source_id for e in r2.output.evidence}
    # Exactly one new BM25 call (C) and one new merge call (D) across the revision.
    assert len(rig.bm25.calls) == 3 and len(rig.merge.calls) == 2
    # The old D is immutable history, unchanged.
    assert old_d is not None and content(old_d) == content(r1.artifacts["D"])


def test_full_and_incremental_execution_give_identical_artifacts(index: BM25Index) -> None:
    q1 = plan_compare(index, "AWS", "Azure")
    q2 = plan_compare(index, "AWS", "GCP")
    incremental, full = make_rig(index), make_rig(index)

    async def scenario() -> tuple[ExecutionResult, ExecutionResult, ExecutionResult]:
        await incremental.executor.execute(q1, principal=PRINCIPAL)
        inc = await incremental.executor.execute(q2, principal=PRINCIPAL)  # reuses A
        ful = await full.executor.execute(
            q2, reuse=False, principal=PRINCIPAL
        )  # fresh store, everything runs
        # Full re-execution on the *populated* store must agree with what is stored,
        # otherwise the store raises DeterminismViolation.
        again = await incremental.executor.execute(q2, reuse=False, principal=PRINCIPAL)
        return inc, ful, again

    inc, ful, again = run(scenario())
    assert inc.reused == {"A"}
    assert ful.executed == again.executed == {"A", "C", "D"}
    for nid in ("A", "C", "D"):
        assert content(inc.artifacts[nid]) == content(ful.artifacts[nid])
        assert content(again.artifacts[nid]) == content(ful.artifacts[nid])
    # Timestamps differ (different clock ticks) yet identity is the same.
    assert inc.artifacts["A"].created_at != again.artifacts["A"].created_at


def test_trace_records_status_per_node(rig: Rig, index: BM25Index) -> None:
    async def scenario() -> ExecutionResult:
        await rig.executor.execute(plan_compare(index, "AWS", "Azure"), principal=PRINCIPAL)
        return await rig.executor.execute(plan_compare(index, "AWS", "GCP"), principal=PRINCIPAL)

    r = run(scenario())
    status = {t.node_id: t.status for t in r.trace}
    assert status == {"A": NodeStatus.REUSED, "C": NodeStatus.EXECUTED, "D": NodeStatus.EXECUTED}
    assert [t.node_id for t in r.trace][-1] == "D"  # topological order
