"""Phase 4: permissions, freshness, eviction, provenance, corruption fallback, epochs,
cited answers, SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from conftest import (
    CORPUS_PATH,
    PRINCIPAL,
    SCOPE,
    Rig,
    TickClock,
    make_policy,
    make_rig,
    plan_hybrid,
    retrieve,
)
from test_acceptance import content

from deltaquery.compiler.compiler import IntentCompiler
from deltaquery.compiler.planner import Planner, PlannerConfig
from deltaquery.evidence.access import AccessDenied, AccessPolicy
from deltaquery.evidence.corpus import Corpus, PassageSet
from deltaquery.evidence.models import Artifact, Evidence
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext
from deltaquery.operators.bm25 import BM25Index, BM25Operator, BM25Params
from deltaquery.operators.dense import DenseIndex, DenseOperator
from deltaquery.operators.fusion import RRFOperator
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.answer import AnswerGenerator, ProvenanceError, SpanResolver
from deltaquery.runtime.executor import ExecutionResult, Executor, NodeStatus
from deltaquery.runtime.models import FreshnessPolicy, Plan
from deltaquery.runtime.validate import ReuseContext, reuse_rejection_reason
from deltaquery.sdk import Runtime
from deltaquery.session import Session, SupersededError, VersionConflict

ENTITIES = {"aws", "azure", "gcp"}
Q1 = {"task": "compare", "entities": ["aws", "azure"], "dimensions": ["disaster_recovery"]}
Q2 = {"task": "compare", "entities": ["aws", "gcp"], "dimensions": ["disaster_recovery"]}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def sources(r: ExecutionResult) -> set[str]:
    return {e.source_id for a in r.artifacts.values() for e in a.evidence}


class ScriptedLLM:
    model_version = "scripted/test@0"

    def __init__(self, outputs: Sequence[dict[str, Any] | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts: list[str] = []

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        self.calls += 1
        self.prompts.append(user)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


# == permissions ==========================================================================


def test_principal_without_scope_gets_nothing(rig: Rig, index: BM25Index) -> None:
    plan = Plan(nodes=(retrieve("A", index, "aws"),), output_id="A")
    with pytest.raises(AccessDenied):
        run(rig.executor.execute(plan, principal="mallory"))
    assert run(rig.store.count_artifacts()) == 0


def test_revoked_principal_cannot_reuse(rig: Rig, index: BM25Index) -> None:
    plan = Plan(nodes=(retrieve("A", index, "aws"),), output_id="A")
    run(rig.executor.execute(plan, principal=PRINCIPAL))
    rig.executor.policy.revoke_scope(PRINCIPAL, SCOPE)
    with pytest.raises(AccessDenied):
        run(rig.executor.execute(plan, principal=PRINCIPAL))


def test_source_revocation_blocks_stale_reuse_and_restoration_reuses(
    rig: Rig, index: BM25Index, dense_index: DenseIndex
) -> None:
    plan = plan_hybrid(index, dense_index, "aws", "gcp")
    policy = rig.executor.policy

    async def go() -> tuple[ExecutionResult, ExecutionResult, ExecutionResult]:
        r1 = await rig.executor.execute(plan, principal=PRINCIPAL)
        policy.revoke_source(SCOPE, "aws-dr-rpo")
        r2 = await rig.executor.execute(plan, principal=PRINCIPAL)
        policy.restore_source(SCOPE, "aws-dr-rpo")
        r3 = await rig.executor.execute(plan, principal=PRINCIPAL)
        return r1, r2, r3

    r1, r2, r3 = run(go())
    assert "aws-dr-rpo" in sources(r1)
    assert "aws-dr-rpo" not in sources(r2)
    # Precise invalidation: only the branch whose candidate domain contains the revoked
    # source (AWS retrievers) and its dependents re-run; the GCP branch is reused.
    assert r2.reused == {"C_lex", "C_dense", "C"}
    assert r2.executed == {"A_lex", "A_dense", "A", "D"}
    assert r3.executed == set()  # same revocations as r1 again
    assert content(r3.output) == content(r1.output)


def test_precise_revocation_matches_full_reexecution(
    index: BM25Index, dense_index: DenseIndex
) -> None:
    """Safety of precise invalidation: after any revocation, incremental results equal a full
    re-execution under the same policy, node by node."""
    plan = plan_hybrid(index, dense_index, "aws", "gcp")
    for revoked in ["aws-dr-rpo", "gcp-dr-overview", "azure-dr-rpo", "aws-pricing"]:
        inc, full = make_rig(index, dense_index), make_rig(index, dense_index)

        async def go(inc: Rig, full: Rig, src: str) -> tuple[ExecutionResult, ExecutionResult]:
            await inc.executor.execute(plan, principal=PRINCIPAL)
            inc.executor.policy.revoke_source(SCOPE, src)
            full.executor.policy.revoke_source(SCOPE, src)
            return (
                await inc.executor.execute(plan, principal=PRINCIPAL),
                await full.executor.execute(plan, principal=PRINCIPAL, reuse=False),
            )

        r_inc, r_full = run(go(inc, full, revoked))
        for n in plan.nodes:
            assert [e.identity() for e in r_inc.artifacts[n.id].evidence] == [
                e.identity() for e in r_full.artifacts[n.id].evidence
            ], (revoked, n.id)


def test_coarse_mode_invalidates_whole_scope(index: BM25Index, dense_index: DenseIndex) -> None:
    rig = make_rig(index, dense_index)
    rig.executor.precise_access = False
    plan = plan_hybrid(index, dense_index, "aws", "gcp")

    async def go() -> ExecutionResult:
        await rig.executor.execute(plan, principal=PRINCIPAL)
        rig.executor.policy.revoke_source(SCOPE, "aws-dr-rpo")
        return await rig.executor.execute(plan, principal=PRINCIPAL)

    assert run(go()).reused == set()


def test_unknown_operators_default_to_all_revocations(index: BM25Index) -> None:
    from deltaquery.operators.base import Operator

    class Custom(BM25Operator):
        access_dependencies = Operator.access_dependencies  # type: ignore[assignment]

    revoked = frozenset({"x", "y"})
    params = BM25Operator(index).parse_params(retrieve("A", index, "aws").canonical_parameters)
    assert Custom(index).access_dependencies(params, SCOPE, revoked) == revoked
    assert BM25Operator(index).access_dependencies(params, SCOPE, revoked) == frozenset()


def test_validator_reauthorizes_even_on_signature_hit(rig: Rig, index: BM25Index) -> None:
    plan = Plan(nodes=(retrieve("A", index, "aws"),), output_id="A")
    r = run(rig.executor.execute(plan, principal=PRINCIPAL))
    art, node = r.output, plan.node("A")
    op = BM25Operator(index)

    def ctx(**kw: Any) -> ReuseContext:
        base: dict[str, Any] = {
            "signature": art.signature,
            "inputs": (),
            "access_policy_version": art.access_policy_version,
            "revoked_sources": frozenset(),
            "principal_may_read": True,
            "now": datetime.now(UTC),
            "last_verified_at": None,
        }
        return ReuseContext(**(base | kw))

    assert reuse_rejection_reason(node, op, art, ctx()) is None
    some_source = art.evidence[0].source_id
    assert (
        reuse_rejection_reason(node, op, art, ctx(revoked_sources=frozenset({some_source})))
        == "revoked_source"
    )
    assert reuse_rejection_reason(node, op, art, ctx(principal_may_read=False)) == (
        "principal_scope"
    )


class LeakyBM25(BM25Operator):
    """Buggy operator that ignores revocations. The executor must refuse its output."""

    async def run(
        self, params: BM25Params, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        leaky = ExecutionContext(ctx.signature, ctx.access_scope, ctx.now, frozenset())
        return await super().run(params, inputs, leaky)


def test_operator_that_leaks_revoked_source_is_refused(index: BM25Index) -> None:
    policy = make_policy()
    store = EvidenceStore()
    ex = Executor(store, [LeakyBM25(index)], policy, clock=TickClock())
    plan = Plan(nodes=(retrieve("A", index, "aws", top_k=20),), output_id="A")
    policy.revoke_source(SCOPE, "aws-dr-rpo")
    with pytest.raises(ExceptionGroup) as info:
        run(ex.execute(plan, principal=PRINCIPAL))
    assert info.group_contains(AccessDenied)
    assert run(store.count_artifacts()) == 0


# == freshness ============================================================================


@dataclass
class SetClock:
    t: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: int) -> None:
        self.t += timedelta(seconds=seconds)


def test_ttl_expires_and_reverification_renews(index: BM25Index) -> None:
    clock = SetClock()
    store = EvidenceStore()
    ex = Executor(store, [BM25Operator(index)], make_policy(), clock=clock)
    node = retrieve("A", index, "aws").model_copy(
        update={"freshness_policy": FreshnessPolicy.TTL, "max_age_s": 60}
    )
    plan = Plan(nodes=(node,), output_id="A")

    def step(advance: int) -> ExecutionResult:
        clock.advance(advance)
        return run(ex.execute(plan, principal=PRINCIPAL))  # type: ignore[no-any-return]

    r0 = step(0)
    r1 = step(30)  # within TTL
    r2 = step(90)  # 120s after compute: expired
    r3 = step(30)  # 30s after re-verification
    assert r0.executed == {"A"} and r1.reused == {"A"}
    assert r2.executed == {"A"}
    assert [t.reuse_rejected for t in r2.trace] == ["freshness"]
    assert r3.reused == {"A"}
    assert content(r2.output) == content(r0.output)


def test_planner_applies_ttl_to_pricing_only(index: BM25Index, dense_index: DenseIndex) -> None:
    from deltaquery.compiler.intent import Intent

    planner = Planner(PlannerConfig(bm25=index, dense=dense_index, access_scope=SCOPE))
    intent = Intent.model_validate(
        Q2 | {"dimensions": ["disaster_recovery", "pricing"]}, context={"entities": ENTITIES}
    )
    plan = planner.build(intent)
    ttl = {n.id for n in plan.nodes if n.freshness_policy is FreshnessPolicy.TTL}
    assert ttl == {f"{e}__pricing__{k}" for e in ("aws", "gcp") for k in ("lex", "dense")}


def test_freshness_requires_max_age(index: BM25Index) -> None:
    from deltaquery.runtime.models import Node

    raw = retrieve("A", index, "aws").model_dump()
    with pytest.raises(ValueError, match="max_age_s"):
        Node.model_validate(raw | {"freshness_policy": "ttl"})
    with pytest.raises(ValueError, match="max_age_s"):
        Node.model_validate(raw | {"max_age_s": 60})


# == eviction and provenance ==============================================================


def test_eviction_is_scoped_and_safe(rig: Rig, index: BM25Index, dense_index: DenseIndex) -> None:
    plan = plan_hybrid(index, dense_index, "aws", "gcp")
    other = Plan(nodes=(retrieve("X", index, "aws", scope="restricted"),), output_id="X")

    async def go() -> tuple[ExecutionResult, int, int, ExecutionResult]:
        r1 = await rig.executor.execute(plan, principal=PRINCIPAL)
        await rig.executor.execute(other, principal=PRINCIPAL)
        ev_before = await rig.store.count_evidence()
        evicted = await rig.store.evict(scope=SCOPE, max_artifacts=3)
        assert await rig.store.count_artifacts(SCOPE) == 3
        assert await rig.store.count_artifacts("restricted") == 1  # other scope untouched
        assert await rig.store.count_evidence() < ev_before  # orphans removed
        r2 = await rig.executor.execute(plan, principal=PRINCIPAL)
        return r1, evicted, len(r2.reused), r2

    r1, evicted, n_reused, r2 = run(go())
    assert evicted == len(plan.nodes) - 3 and n_reused >= 1 and r2.executed
    for n in plan.nodes:  # recomputation after eviction gives identical artifacts
        assert content(r2.artifacts[n.id]) == content(r1.artifacts[n.id])


def test_lineage_and_evidence_lookup(rig: Rig, index: BM25Index, dense_index: DenseIndex) -> None:
    plan = plan_hybrid(index, dense_index, "aws", "gcp")
    r = run(rig.executor.execute(plan, principal=PRINCIPAL))
    lineage = run(rig.store.lineage(r.output.signature))
    assert set(lineage) == {a.signature for a in r.artifacts.values()}
    assert list(lineage)[-1] == r.output.signature  # inputs first
    e = r.output.evidence[0]
    assert run(rig.store.get_evidence(e.evidence_id)).identity() == e.identity()
    # Evict everything but the output: lineage reports the gaps instead of failing.
    run(rig.store.evict(scope=SCOPE, max_artifacts=0))
    assert run(rig.store.lineage(r.output.signature)) == {r.output.signature: None}


def test_corrupt_artifact_falls_back_to_recompute(rig: Rig, index: BM25Index) -> None:
    plan = Plan(nodes=(retrieve("A", index, "aws"),), output_id="A")
    r1 = run(rig.executor.execute(plan, principal=PRINCIPAL))
    conn = rig.store._raw_connection_for_tests()
    conn.execute("DROP TRIGGER evidence_immutable")
    conn.execute("UPDATE evidence SET span_end = span_end + 1")  # simulate disk corruption
    # A validated in-process copy is still served (disk corruption cannot reach it)...
    assert run(rig.executor.execute(plan, principal=PRINCIPAL)).reused == {"A"}
    # ...but after a restart the corrupt row is detected and recomputed.
    rig.store.clear_memory_cache()
    r2 = run(rig.executor.execute(plan, principal=PRINCIPAL))
    assert r2.executed == {"A"} and [t.reuse_rejected for t in r2.trace] == ["corrupt"]
    assert content(r2.output) == content(r1.output)


# == epochs ===============================================================================


class GatedLLM:
    """First call blocks on a gate; later calls answer immediately.
    `cooperative=False` swallows cancellation (simulates work that cannot be interrupted)."""

    model_version = "gated/test@0"

    def __init__(self, outputs: list[dict[str, Any]], *, cooperative: bool) -> None:
        self.outputs = outputs
        self.cooperative = cooperative
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.n = 0

    async def complete_json(self, **_: Any) -> dict[str, Any]:
        i = self.n
        self.n += 1
        if i == 0:
            self.entered.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                if self.cooperative:
                    raise
                await self.gate.wait()
        return self.outputs[i]


def _session(index: BM25Index, dense_index: DenseIndex, llm: Any) -> Session:
    ops = [BM25Operator(index), DenseOperator(dense_index), RRFOperator(), MergeOperator()]
    ex = Executor(EvidenceStore(), ops, make_policy(), clock=TickClock())
    planner = Planner(PlannerConfig(bm25=index, dense=dense_index, access_scope=SCOPE))
    return Session(IntentCompiler(llm, ENTITIES), planner, ex, principal=PRINCIPAL)


@pytest.mark.parametrize("cooperative", [True, False])
def test_superseded_request_never_commits(
    index: BM25Index, dense_index: DenseIndex, cooperative: bool
) -> None:
    llm = GatedLLM([Q1, Q2], cooperative=cooperative)
    session = _session(index, dense_index, llm)

    async def go() -> tuple[BaseException | None, Any]:
        first = asyncio.create_task(session.ask("Compare AWS and Azure DR"))
        await llm.entered.wait()
        second = await session.ask("Compare AWS and GCP DR")
        llm.gate.set()  # let the old work finish (matters when it ignored cancellation)
        try:
            await first
            err: BaseException | None = None
        except SupersededError as exc:
            err = exc
        return err, second

    err, second = run(go())
    assert isinstance(err, SupersededError)
    assert session.turns == [second] and session.version == 1
    assert second.intent.entities == ("aws", "gcp")


def test_expected_version_conflict(index: BM25Index, dense_index: DenseIndex) -> None:
    llm = ScriptedLLM([Q1])
    session = _session(index, dense_index, llm)
    run(session.ask("Compare AWS and Azure DR", expected_version=0))
    with pytest.raises(VersionConflict):
        run(session.ask("Replace Azure with GCP", expected_version=0))
    assert llm.calls == 1 and session.version == 1


# == answers ==============================================================================


def _hybrid_output(rig: Rig, index: BM25Index, dense_index: DenseIndex) -> tuple[Evidence, ...]:
    r = run(
        rig.executor.execute(plan_hybrid(index, dense_index, "aws", "gcp"), principal=PRINCIPAL)
    )
    return r.output.evidence  # type: ignore[no-any-return]


def test_answer_citations_carry_verified_quotes(
    rig: Rig, index: BM25Index, dense_index: DenseIndex, passages: PassageSet
) -> None:
    ev = _hybrid_output(rig, index, dense_index)
    llm = ScriptedLLM([{"abstain": False, "sentences": [{"text": "X.", "citations": [2, 1, 2]}]}])
    ans = run(AnswerGenerator(llm, SpanResolver(passages)).generate("q", ev))
    assert not ans.abstained
    (s,) = ans.sentences
    assert [c.ref for c in s.citations] == [1, 2]
    by_key = {(p.source_id, p.span): p.text for p in passages.passages}
    for c, e in zip(s.citations, ev[:2], strict=True):
        assert c.evidence_id == e.evidence_id and c.quote == by_key[(e.source_id, e.span)]
    # Evidence is presented as delimited, quoted data.
    assert llm.prompts[0].count("<passage ") == min(len(ev), 12)


@pytest.mark.parametrize(
    "outputs",
    [
        [{"abstain": False, "sentences": [{"text": "X.", "citations": [99]}]}] * 2,
        [{"abstain": False, "sentences": [{"text": "X.", "citations": []}]}] * 2,
        [{"abstain": False, "sentences": []}] * 2,
        [{"abstain": True, "sentences": []}],
    ],
)
def test_answer_abstains_on_unvalidated_or_declined_output(
    rig: Rig,
    index: BM25Index,
    dense_index: DenseIndex,
    passages: PassageSet,
    outputs: list[dict[str, Any]],
) -> None:
    ev = _hybrid_output(rig, index, dense_index)
    ans = run(AnswerGenerator(ScriptedLLM(outputs), SpanResolver(passages)).generate("q", ev))
    assert ans.abstained and ans.sentences == ()


def test_answer_abstains_without_evidence_and_without_calling_llm(passages: PassageSet) -> None:
    llm = ScriptedLLM([])
    ans = run(AnswerGenerator(llm, SpanResolver(passages)).generate("q", ()))
    assert ans.abstained and ans.reason == "no evidence" and llm.calls == 0


def test_answer_rejects_unverifiable_evidence(
    rig: Rig, index: BM25Index, dense_index: DenseIndex, passages: PassageSet
) -> None:
    ev = _hybrid_output(rig, index, dense_index)
    forged = ev[0].model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(ProvenanceError):
        run(AnswerGenerator(ScriptedLLM([]), SpanResolver(passages)).generate("q", (forged,)))


# == SDK ==================================================================================


def test_sdk_session_flow() -> None:
    answer = {"abstain": False, "sentences": [{"text": "See evidence.", "citations": [1]}]}
    llm = ScriptedLLM([Q1, answer, Q2, answer])
    policy = AccessPolicy()
    policy.grant("alice", SCOPE)

    async def go() -> tuple[Any, Any]:
        runtime = await Runtime.create(
            corpus=Corpus.from_json(CORPUS_PATH), llm=llm, policy=policy, entities=ENTITIES
        )
        with pytest.raises(AccessDenied):
            runtime.session(scope=SCOPE, principal="bob")
        session = runtime.session(scope=SCOPE, principal="alice")
        first = await session.ask("Compare AWS and Azure DR")
        nxt = await session.ask("Replace Azure with GCP")
        await runtime.close()
        return first, nxt

    first, nxt = run(go())
    assert first.trace.reused_nodes == ()
    assert set(nxt.trace.reused_nodes) >= {"aws__disaster_recovery", "aws__evidence"}
    assert "azure__evidence" in nxt.trace.removed_nodes
    assert nxt.answer is not None and not nxt.answer.abstained
    # The generator gets a self-contained question rendered from the intent, not just the
    # revision text.
    assert llm.prompts[3].startswith(
        "QUESTION: Compare aws, gcp on: disaster recovery. (Latest message: Replace Azure with GCP)"
    )
    d = json.loads(json.dumps(nxt.trace.to_dict()))
    assert d["version"] == 2 and set(d["timing_s"]) == {"compile", "retrieval", "generation"}
    assert all(v >= 0 for v in d["timing_s"].values())


def test_nodes_status_enum_is_stable() -> None:
    # The trace format is consumed by the (upcoming) UI; keep the wire values fixed.
    assert [s.value for s in NodeStatus] == ["executed", "reused"]


def test_restoring_a_source_recomputes_affected_branch(
    index: BM25Index, dense_index: DenseIndex
) -> None:
    """The other direction: an artifact computed while a source was revoked must not be
    reused after the source is restored (it would silently miss evidence)."""
    plan = plan_hybrid(index, dense_index, "aws", "gcp")
    inc, full = make_rig(index, dense_index), make_rig(index, dense_index)

    async def go() -> tuple[ExecutionResult, ExecutionResult]:
        inc.executor.policy.revoke_source(SCOPE, "aws-dr-rpo")
        await inc.executor.execute(plan, principal=PRINCIPAL)
        inc.executor.policy.restore_source(SCOPE, "aws-dr-rpo")
        return (
            await inc.executor.execute(plan, principal=PRINCIPAL),
            await full.executor.execute(plan, principal=PRINCIPAL, reuse=False),
        )

    r_inc, r_full = run(go())
    assert {"A_lex", "A_dense"} <= r_inc.executed
    assert content(r_inc.output) == content(r_full.output)
    assert "aws-dr-rpo" in sources(r_inc)
