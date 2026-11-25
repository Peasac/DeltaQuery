"""Phase 2: dense adapter, shared span format, fusion, versioned snapshots."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import (
    CORPUS_PATH,
    PRINCIPAL,
    SCOPE,
    Rig,
    TickClock,
    dense,
    make_policy,
    make_rig,
    plan_hybrid,
)
from test_acceptance import content

from deltaquery.evidence.corpus import Corpus, PassageSet
from deltaquery.evidence.models import Artifact, CoverageType, Evidence, Span, compute_artifact_hash
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext
from deltaquery.operators.bm25 import BM25Index, BM25Operator
from deltaquery.operators.dense import DenseIndex, DenseOperator
from deltaquery.operators.embedding import EmbeddingError, HashingEmbedder
from deltaquery.operators.fusion import FusionInputError, RRFOperator, RRFParams
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.executor import ExecutionResult, Executor
from deltaquery.runtime.models import Plan, PlanValidationError

RETRIEVERS = {"A_lex", "A_dense"}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def twice(rig: Rig, p1: Plan, p2: Plan) -> tuple[ExecutionResult, ExecutionResult]:
    async def go() -> tuple[ExecutionResult, ExecutionResult]:
        return await rig.executor.execute(p1, principal=PRINCIPAL), await rig.executor.execute(
            p2, principal=PRINCIPAL
        )

    return run(go())  # type: ignore[no-any-return]


# -- shared span format -------------------------------------------------------------------


def test_lexical_and_dense_evidence_share_span_format(
    rig: Rig, index: BM25Index, dense_index: DenseIndex, passages: PassageSet
) -> None:
    r = run(
        rig.executor.execute(plan_hybrid(index, dense_index, "aws", "azure"), principal=PRINCIPAL)
    )
    lex, den = r.artifacts["A_lex"].evidence, r.artifacts["A_dense"].evidence
    assert {e.coverage_type for e in lex} == {CoverageType.LEXICAL}
    assert {e.coverage_type for e in den} == {CoverageType.DENSE}
    known = {(p.source_id, p.source_version, p.span, p.content_hash) for p in passages.passages}

    def fmt(e: Evidence) -> tuple[str, str, Span, str]:
        return (e.source_id, e.source_version, e.span, e.content_hash)

    assert {fmt(e) for e in lex + den} <= known
    shared = {fmt(e) for e in lex} & {fmt(e) for e in den}
    assert shared, "fixture should make both retrievers find at least one common passage"


# -- reuse on hybrid plans ----------------------------------------------------------------


def test_hybrid_identical_query_reuses_everything(
    rig: Rig, index: BM25Index, dense_index: DenseIndex
) -> None:
    q = plan_hybrid(index, dense_index, "aws", "azure")
    _r1, r2 = twice(rig, q, q)
    assert r2.executed == set() and r2.reused == {n.id for n in q.nodes}
    assert rig.dense is not None and len(rig.dense.calls) == 2


def test_hybrid_entity_change_keeps_untouched_branch(
    rig: Rig, index: BM25Index, dense_index: DenseIndex
) -> None:
    _r1, r2 = twice(
        rig,
        plan_hybrid(index, dense_index, "aws", "azure"),
        plan_hybrid(index, dense_index, "aws", "gcp"),
    )
    assert r2.reused == {"A_lex", "A_dense", "A"}
    assert r2.executed == {"C_lex", "C_dense", "C", "D"}


def test_changed_fusion_recomputes_fusion_only(
    rig: Rig, index: BM25Index, dense_index: DenseIndex
) -> None:
    r1, r2 = twice(
        rig,
        plan_hybrid(index, dense_index, "aws", "azure", k=60),
        plan_hybrid(index, dense_index, "aws", "azure", k=10),
    )
    assert r2.reused == {"A_lex", "A_dense", "B_lex", "B_dense"}
    assert {"A", "B"} <= r2.executed
    # The fused artifact was computed over the *current* candidate sets (same signatures).
    assert r2.artifacts["A"].input_signatures == r1.artifacts["A"].input_signatures


def test_hybrid_full_equals_incremental(index: BM25Index, dense_index: DenseIndex) -> None:
    q1 = plan_hybrid(index, dense_index, "aws", "azure")
    q2 = plan_hybrid(index, dense_index, "aws", "gcp")
    inc_rig, full_rig = make_rig(index, dense_index), make_rig(index, dense_index)

    async def go() -> tuple[ExecutionResult, ExecutionResult, ExecutionResult]:
        await inc_rig.executor.execute(q1, principal=PRINCIPAL)
        inc = await inc_rig.executor.execute(q2, principal=PRINCIPAL)
        ful = await full_rig.executor.execute(q2, reuse=False, principal=PRINCIPAL)
        again = await inc_rig.executor.execute(
            q2, reuse=False, principal=PRINCIPAL
        )  # DeterminismViolation if not
        return inc, ful, again

    inc, ful, again = run(go())
    assert inc.reused == {"A_lex", "A_dense", "A"}
    for n in q2.nodes:
        assert content(inc.artifacts[n.id]) == content(ful.artifacts[n.id])
        assert content(again.artifacts[n.id]) == content(ful.artifacts[n.id])


# -- dense canonicalization and model versioning ------------------------------------------


def test_dense_query_canonicalization_is_conservative(
    rig: Rig, index: BM25Index, dense_index: DenseIndex
) -> None:
    def p(q: str) -> Plan:
        return Plan(nodes=(dense("X", dense_index, "aws", query=q),), output_id="X")

    async def go() -> tuple[ExecutionResult, ...]:
        a = await rig.executor.execute(p("disaster  recovery\nRPO"), principal=PRINCIPAL)
        b = await rig.executor.execute(
            p(" disaster recovery RPO "), principal=PRINCIPAL
        )  # whitespace only
        c = await rig.executor.execute(
            p("Disaster Recovery RPO"), principal=PRINCIPAL
        )  # case: not assumed equal
        return a, b, c

    a, b, c = run(go())
    assert a.executed == {"X"} and b.reused == {"X"} and c.executed == {"X"}


def test_changed_embedding_model_does_not_reuse(
    index: BM25Index, dense_index: DenseIndex, passages: PassageSet
) -> None:
    other = run(DenseIndex.build(passages, HashingEmbedder(dim=128)))
    assert other.model_version != dense_index.model_version
    rig = make_rig(index, dense_index, DenseOperator(other))
    old = Plan(nodes=(dense("X", dense_index, "aws"),), output_id="X")
    new = Plan(nodes=(dense("X", other, "aws"),), output_id="X")
    r1, r2 = twice(rig, old, new)
    assert r2.executed == {"X"}
    assert r2.output.model_version == other.model_version != r1.output.model_version


def test_embedder_output_is_validated(passages: PassageSet) -> None:
    class Broken(HashingEmbedder):
        async def embed(self, texts: Any) -> list[tuple[float, ...]]:
            return [(1.0, 2.0)] * len(texts)

    with pytest.raises(EmbeddingError):
        run(DenseIndex.build(passages, Broken()))


# -- fusion unit behaviour ----------------------------------------------------------------

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _ev(sid: str, start: int, version: str = "v1") -> Evidence:
    return Evidence(
        source_id=sid,
        source_version=version,
        span=Span(start=start, end=start + 10),
        content_hash=f"h-{sid}-{start}",
        retrieval_node_id="r",
        score=1.0,
        coverage_type=CoverageType.LEXICAL,
        created_at=_T,
    )


def _art(sig: str, *ev: Evidence, contract: str = "ranked_passages") -> Artifact:
    return Artifact(
        signature=sig,
        operator="bm25",
        operator_version="1",
        output_contract=contract,
        access_scope=SCOPE,
        access_policy_version="pv",
        source_snapshot="s",
        index_version="i",
        model_version=None,
        input_signatures=(),
        evidence=ev,
        artifact_hash=compute_artifact_hash(contract, ev),
        created_at=_T,
    )


def test_rrf_scores_and_order() -> None:
    a, b, c = _ev("a", 0), _ev("b", 0), _ev("c", 0)
    ctx = ExecutionContext(signature="F", access_scope=SCOPE, now=_T)
    out = run(RRFOperator().run(RRFParams(k=1, top_k=3), [_art("x", a, b), _art("y", b, c)], ctx))
    # b: 1/(1+2) + 1/(1+1); a: 1/(1+1); c: 1/(1+2)
    assert [e.source_id for e in out] == ["b", "a", "c"]
    assert out[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert {e.coverage_type for e in out} == {CoverageType.FUSED}
    assert {e.retrieval_node_id for e in out} == {"F"}


def test_rrf_rejects_span_mismatch_and_wrong_contract() -> None:
    ctx = ExecutionContext(signature="F", access_scope=SCOPE, now=_T)
    op, params = RRFOperator(), RRFParams(top_k=3)
    with pytest.raises(FusionInputError, match="span format"):
        run(op.run(params, [_art("x", _ev("a", 0)), _art("y", _ev("a", 0, "v2"))], ctx))
    with pytest.raises(FusionInputError, match="ranked_passages"):
        run(op.run(params, [_art("x", _ev("a", 0)), _art("y", contract="evidence_set")], ctx))


# -- versioned snapshots / source update --------------------------------------------------


def _updated_corpus() -> Corpus:
    docs = [
        d.model_copy(update={"text": d.text + "\n\nRPO is now documented as five seconds."})
        if d.source_id == "gcp-dr-rpo"
        else d
        for d in Corpus.from_json(CORPUS_PATH).documents
    ]
    return Corpus(docs)


def test_source_update_blocks_stale_reuse_and_keeps_old_snapshot(
    index: BM25Index, dense_index: DenseIndex
) -> None:
    ps2 = PassageSet(_updated_corpus())
    assert ps2.source_snapshot != index.snapshot_id
    index2 = BM25Index(ps2)
    dense2 = run(DenseIndex.build(ps2, HashingEmbedder()))
    store = EvidenceStore(":memory:")
    ex = Executor(
        store,
        [
            BM25Operator(index),
            DenseOperator(dense_index),
            BM25Operator(index2),
            DenseOperator(dense2),
            RRFOperator(),
            MergeOperator(),
        ],
        make_policy(),
        clock=TickClock(),
    )
    old_q = plan_hybrid(index, dense_index, "aws", "gcp")
    new_q = plan_hybrid(index2, dense2, "aws", "gcp")

    async def go() -> tuple[ExecutionResult, ExecutionResult, ExecutionResult]:
        return (
            await ex.execute(old_q, principal=PRINCIPAL),
            await ex.execute(new_q, principal=PRINCIPAL),
            await ex.execute(old_q, principal=PRINCIPAL),
        )

    r_old, r_new, r_old_again = run(go())
    retrievers = {"A_lex", "A_dense", "C_lex", "C_dense"}
    # Snapshot-level invalidation: every retriever re-runs on the new snapshot, even AWS
    # whose documents did not change (conservative; finer-grained invalidation is later work).
    assert retrievers <= r_new.executed
    gcp_old = {e.source_version for e in r_old.artifacts["C_lex"].evidence}
    gcp_new = {e.source_version for e in r_new.artifacts["C_lex"].evidence}
    assert gcp_old != gcp_new
    # The old snapshot is still registered, so its artifacts remain valid for old plans.
    assert r_old_again.executed == set()


def test_plan_for_unregistered_snapshot_fails_closed(
    index: BM25Index, dense_index: DenseIndex
) -> None:
    ps2 = PassageSet(_updated_corpus())
    rig = make_rig(BM25Index(ps2), run(DenseIndex.build(ps2, HashingEmbedder())))
    with pytest.raises(PlanValidationError, match="versions"):
        run(
            rig.executor.execute(plan_hybrid(index, dense_index, "aws", "gcp"), principal=PRINCIPAL)
        )


def test_duplicate_operator_registration_rejected(index: BM25Index) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Executor(EvidenceStore(), [BM25Operator(index), BM25Operator(index)], make_policy())
