"""Evidence sufficiency: claim extraction, gated retrieval, safety under revision/revocation.

Uses the synthetic DeltaBench world, whose documents state exact values ("RPO is 30 seconds").
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deltaquery.compiler.intent import Intent
from deltaquery.compiler.planner import Planner, PlannerConfig
from deltaquery.eval.synth import SCOPE, World, build_world, gold_keys, source_id
from deltaquery.eval.systems import PRINCIPAL, Env, new_policy
from deltaquery.evidence.access import AccessPolicy
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.claims import extract_claims
from deltaquery.runtime.executor import ExecutionResult, Executor, NodeStatus


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def world() -> World:
    return build_world(7)


@pytest.fixture(scope="module")
def env(world: World) -> Env:
    return run(Env.create({0: world.corpus(0)}, SCOPE))  # type: ignore[no-any-return]


def planner(env: Env, sufficiency: bool) -> Planner:
    s = env.snapshots[0]
    return Planner(
        PlannerConfig(bm25=s.bm25, dense=s.dense, access_scope=SCOPE, sufficiency=sufficiency)
    )


def intent(entities: list[str], dims: list[str], rpo: bool = True) -> Intent:
    cons = [{"attribute": "rpo", "op": "eq", "value_seconds": 0}] if rpo else []
    return Intent.model_validate(
        {"task": "compare", "entities": entities, "dimensions": dims, "constraints": cons}
    )


def covered_entities(world: World, env: Env) -> list[str]:
    """Entities whose DR branch alone surfaces their exact-RPO paragraph (so the check can
    find a claim). Computed, not assumed: with fused_k=3 the DR branch sometimes ranks the
    pricing paragraph above the RPO paragraph, and then coverage is genuinely absent."""
    from deltaquery.compiler.intent import Dimension

    out = []
    p = planner(env, False)
    for e in world.entity_ids:
        it = Intent.model_validate(
            {"task": "describe", "entities": [e], "dimensions": ["disaster_recovery"]}
        )
        r = Rig(env).go(p, it)
        sid = source_id(world.name_of(e), Dimension.DISASTER_RECOVERY)
        texts = {
            q.text
            for q in env.snapshots[0].passages.passages
            for ev in r.artifacts[f"{e}__disaster_recovery"].evidence
            if (q.source_id, q.span.start) == (ev.source_id, ev.span.start) and q.source_id == sid
        }
        if any(extract_claims(t, ("rpo",)) for t in texts):
            out.append(e)
    return out


def _text(env: Env, key: tuple[str, int, int]) -> str:
    for q in env.snapshots[0].passages.passages:
        if (q.source_id, q.span.start, q.span.end) == key:
            return q.text
    raise KeyError(key)


def keys(r: ExecutionResult) -> set[tuple[str, int, int]]:
    return {(e.source_id, e.span.start, e.span.end) for e in r.output.evidence}


class Rig:
    def __init__(self, env: Env, policy: AccessPolicy | None = None) -> None:
        self.policy = policy or new_policy(SCOPE)
        self.ex = Executor(EvidenceStore(), env.operators(), self.policy)

    def go(self, p: Planner, it: Intent, reuse: bool = True) -> ExecutionResult:
        return run(self.ex.execute(p.build(it), principal=PRINCIPAL, reuse=reuse))  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("the Kel RPO is 30 seconds of data loss", ("rpo=30s",)),
        ("RPO is 2 minutes", ("rpo=120s",)),
        ("RPO is typically under one minute", ()),
        ("an RPO of about 30 seconds", ()),
        ("RPO is less than 5 seconds", ()),
        ("Recovery point objective (RPO) is the maximum acceptable data loss.", ()),
        ("RTO: restores service within 900 seconds", ()),  # wrong attribute requested
    ],
)
def test_extractor_is_exact_and_conservative(text: str, want: tuple[str, ...]) -> None:
    assert extract_claims(text, ("rpo",)) == want


def test_sufficient_evidence_skips_retrieval_at_equal_recall(world: World, env: Env) -> None:
    covered = covered_entities(world, env)
    assert len(covered) >= 2, "fixture must contain covered entities"
    e1, e2 = covered[:2]
    it = intent([e1, e2], ["disaster_recovery"])
    with_s = Rig(env).go(planner(env, True), it)
    without = Rig(env).go(planner(env, False), it)
    skipped = with_s.ids_with(NodeStatus.SKIPPED)
    assert {f"{e}__rpo{s}" for e in (e1, e2) for s in ("", "__lex", "__dense")} <= skipped
    claims = {c for e in with_s.artifacts[f"{e1}__rpo__check"].evidence for c in e.extracted_claims}
    assert claims == {f"rpo={world.facts[0][e1].rpo}s"}
    # Guaranteed: the constraint's own gold passage (the exact RPO statement) is present.
    rpo_only = Intent.model_validate(
        {"task": "compare", "entities": [e1, e2], "dimensions": ["disaster_recovery"]}
    )
    dr_gold = gold_keys(world, env.snapshots[0].corpus, rpo_only, frozenset())
    rpo_gold = {k for k in dr_gold if extract_claims(_text(env, k), ("rpo",))}
    assert len(rpo_gold) == 2 and rpo_gold <= keys(with_s)
    # Not guaranteed: skipping retrieval can lose passages the targeted retrieval would have
    # found incidentally (here the RTO paragraph). The benchmark measures this trade-off.
    gold = gold_keys(world, env.snapshots[0].corpus, it, frozenset())
    assert gold <= keys(without)
    assert len(gold & keys(with_s)) <= len(gold & keys(without))


def test_insufficient_evidence_retrieves(world: World, env: Env) -> None:
    e1, e2 = world.entity_ids[:2]
    it = intent([e1, e2], ["pricing"])  # pricing evidence states no RPO
    r = Rig(env).go(planner(env, True), it)
    assert not r.ids_with(NodeStatus.SKIPPED)
    assert f"{e1}__rpo__lex" in r.executed
    gold = gold_keys(world, env.snapshots[0].corpus, it, frozenset())
    assert gold <= keys(r)


def test_gated_plans_full_equals_incremental(world: World, env: Env) -> None:
    e1, e2 = world.entity_ids[:2]
    p = planner(env, True)
    steps = [
        intent([e1, e2], ["disaster_recovery"], rpo=False),
        intent([e1, e2], ["disaster_recovery"]),
        intent([e1, e2], ["pricing"]),
        intent([e1, e2], ["disaster_recovery", "pricing"]),
    ]
    inc = Rig(env)
    for it in steps:
        r_inc = inc.go(p, it)
        r_full = Rig(env).go(p, it, reuse=False)
        for n in p.build(it).nodes:
            a, b = r_inc.artifacts[n.id], r_full.artifacts[n.id]
            assert [e.identity() for e in a.evidence] == [e.identity() for e in b.evidence]
        assert r_inc.ids_with(NodeStatus.SKIPPED) == r_full.ids_with(NodeStatus.SKIPPED)


def test_revoking_the_covering_source_flips_the_gate(world: World, env: Env) -> None:
    e1, e2 = covered_entities(world, env)[:2]
    p = planner(env, True)
    it = intent([e1, e2], ["disaster_recovery"])
    rig = Rig(env)
    first = rig.go(p, it)
    assert f"{e1}__rpo__lex" in first.ids_with(NodeStatus.SKIPPED)
    dr_doc = source_id(world.name_of(e1), next(iter(Intent.model_validate(
        {"task": "describe", "entities": [e1], "dimensions": ["disaster_recovery"]}
    ).dimensions)))  # fmt: skip
    rig.policy.revoke_source(SCOPE, dr_doc)
    after = rig.go(p, it)
    assert f"{e1}__rpo__lex" in after.executed  # coverage gone -> retrieval runs
    assert f"{e2}__rpo__lex" in after.ids_with(NodeStatus.SKIPPED)  # untouched entity
    assert dr_doc not in {e.source_id for e in after.output.evidence}
    full_policy = new_policy(SCOPE)
    full_policy.revoke_source(SCOPE, dr_doc)
    full = Rig(env, full_policy).go(p, it, reuse=False)
    assert keys(after) == keys(full)
