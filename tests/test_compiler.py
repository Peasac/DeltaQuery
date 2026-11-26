"""Phase 3: intent schema, revision diff, compiler validation, template planner, and the §5
worked example end-to-end with recorded LLM outputs (no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import PRINCIPAL, SCOPE, TickClock, make_policy
from pydantic import ValidationError
from test_acceptance import content

from deltaquery.compiler.compiler import CompileError, IntentCompiler
from deltaquery.compiler.diff import RevisionKind, diff_intents
from deltaquery.compiler.intent import Intent, intent_json_schema
from deltaquery.compiler.planner import OUTPUT_ID, Planner, PlannerConfig
from deltaquery.evidence.store import EvidenceStore
from deltaquery.llm.base import LLMError
from deltaquery.operators.bm25 import BM25Index, BM25Operator
from deltaquery.operators.coverage import ClaimCheckOperator
from deltaquery.operators.dense import DenseIndex, DenseOperator
from deltaquery.operators.fusion import RRFOperator
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.executor import Executor
from deltaquery.session import Session, Turn

ENTITIES = {"aws", "azure", "gcp"}
RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded_intents.json").read_text(encoding="utf-8")
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class ScriptedLLM:
    """Returns queued outputs in order; records every prompt it was given."""

    model_version = "scripted/test@0"

    def __init__(self, outputs: list[dict[str, Any] | Exception]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[tuple[str, str]] = []

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        self.prompts.append((system, user))
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def I(**kw: Any) -> Intent:  # noqa: E743
    base: dict[str, Any] = {"task": "compare", "dimensions": ["disaster_recovery"]}
    return Intent.model_validate(base | kw, context={"entities": ENTITIES})


# -- intent -------------------------------------------------------------------------------


def test_intent_is_canonical() -> None:
    a = I(entities=["Azure", "AWS"], dimensions=["pricing", "disaster_recovery", "pricing"])
    b = I(entities=["aws", "azure"], dimensions=["disaster_recovery", "pricing"])
    assert a == b and a.entities == ("aws", "azure")


@pytest.mark.parametrize(
    "bad",
    [
        {"entities": ["aws", "oracle"]},  # outside vocabulary
        {"entities": ["aws"]},  # compare needs two
        {"entities": ["aws", "gcp"], "dimensions": []},
        {"entities": ["aws", "gcp"], "extra": 1},
        {
            "entities": ["aws", "gcp"],
            "constraints": [{"attribute": "rpo", "op": "eq", "value_seconds": -1}],
        },
    ],
)
def test_invalid_intents_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        I(**bad)


def test_schema_constrains_entities() -> None:
    schema = intent_json_schema(ENTITIES)
    assert schema["properties"]["entities"]["items"] == {"enum": ["aws", "azure", "gcp"]}
    assert schema["additionalProperties"] is False


# -- diff ---------------------------------------------------------------------------------


def test_diff_classifies_revisions() -> None:
    q1 = I(entities=["aws", "azure"])
    q2 = I(entities=["aws", "gcp"])
    q3 = I(
        entities=["aws", "gcp"], constraints=[{"attribute": "rpo", "op": "eq", "value_seconds": 0}]
    )
    q4 = q3.model_copy(
        update={
            "dimensions": I(
                entities=["aws", "gcp"], dimensions=["disaster_recovery", "pricing"]
            ).dimensions
        }
    )
    kinds = [
        [o.kind for o in diff_intents(p, n)]
        for p, n in [(None, q1), (q1, q2), (q2, q3), (q3, q4), (q4, q4)]
    ]
    assert kinds == [
        [RevisionKind.NEW],
        [RevisionKind.REPLACE_ENTITY],
        [RevisionKind.ADD_CONSTRAINT],
        [RevisionKind.ADD_DIMENSION],
        [RevisionKind.UNCHANGED],
    ]
    (op,) = diff_intents(q1, q2)
    assert (op.old, op.new) == ("azure", "gcp")


# -- compiler -----------------------------------------------------------------------------


def test_compiler_prompt_contains_only_user_text_and_prior_intent() -> None:
    llm = ScriptedLLM([RECORDED["sequence"][1]["output"]])
    c = IntentCompiler(llm, ENTITIES)
    prior = I(entities=["aws", "azure"])
    r = run(c.compile("Replace Azure with GCP", prior))
    assert r.intent.entities == ("aws", "gcp") and r.attempts == 1
    ((_, user),) = llm.prompts
    assert user.splitlines() == [
        "Allowed entities: aws, azure, gcp",
        "PREVIOUS INTENT: " + json.dumps(prior.model_dump(mode="json"), sort_keys=True),
        "NEW MESSAGE: Replace Azure with GCP",
    ]


def test_compiler_retries_then_fails_closed() -> None:
    good = RECORDED["sequence"][0]["output"]
    bad = {"task": "compare", "entities": ["aws", "oracle"], "dimensions": ["pricing"]}
    q = "Compare AWS and Azure disaster recovery"
    r = run(IntentCompiler(ScriptedLLM([bad, good]), ENTITIES).compile(q))
    assert r.attempts == 2
    with pytest.raises(CompileError, match="oracle"):
        run(IntentCompiler(ScriptedLLM([bad, LLMError("boom")]), ENTITIES).compile(q))


# -- planner + session: the §5 worked example ---------------------------------------------


def _session(index: BM25Index, dense_index: DenseIndex, llm: Any) -> Session:
    store = EvidenceStore()
    ops = [
        BM25Operator(index),
        DenseOperator(dense_index),
        RRFOperator(),
        MergeOperator(),
        ClaimCheckOperator(index.passages),
    ]
    planner = Planner(PlannerConfig(bm25=index, dense=dense_index, access_scope=SCOPE))
    executor = Executor(store, ops, make_policy(), clock=TickClock())
    return Session(IntentCompiler(llm, ENTITIES), planner, executor, principal=PRINCIPAL)


def _branch(entity: str, slot: str) -> set[str]:
    base = f"{entity}__{slot}"
    return {f"{base}__lex", f"{base}__dense", base}


def test_planner_is_order_independent(index: BM25Index, dense_index: DenseIndex) -> None:
    p = Planner(PlannerConfig(bm25=index, dense=dense_index, access_scope=SCOPE))
    assert p.build(I(entities=["gcp", "aws"])) == p.build(I(entities=["aws", "gcp"]))


def test_worked_example_q1_to_q4(index: BM25Index, dense_index: DenseIndex) -> None:
    seq = RECORDED["sequence"]
    session = _session(index, dense_index, ScriptedLLM([s["output"] for s in seq]))

    async def go() -> list[Turn]:
        return [await session.ask(s["text"]) for s in seq]

    t1, t2, t3, t4 = run(go())
    aws_dr, gcp_dr = _branch("aws", "disaster_recovery"), _branch("gcp", "disaster_recovery")

    # Q1: everything executes.
    assert t1.result.reused == set()

    # Q2 replace(Azure, GCP): reuse AWS, remove Azure, execute GCP, recompute joins.
    assert [o.kind for o in t2.compiled.ops] == [RevisionKind.REPLACE_ENTITY]
    assert t2.result.reused == aws_dr | {"aws__evidence"}
    assert t2.result.executed == gcp_dr | {"gcp__evidence", OUTPUT_ID}
    assert t2.removed == _branch("azure", "disaster_recovery") | {"azure__evidence"}
    assert not any(e.source_id.startswith("azure") for e in t2.result.output.evidence)

    # Q3 add_constraint(RPO == 0): DR evidence kept; sufficiency check over it. In this
    # fixture neither provider states an exact RPO ("typically under one minute", "of zero"
    # with no unit), so the checks find nothing and targeted retrieval runs.
    checks = {"aws__rpo__check", "gcp__rpo__check"}
    assert [o.kind for o in t3.compiled.ops] == [RevisionKind.ADD_CONSTRAINT]
    assert t3.result.reused == aws_dr | gcp_dr
    assert t3.result.executed == (
        checks
        | _branch("aws", "rpo")
        | _branch("gcp", "rpo")
        | {"aws__evidence", "gcp__evidence", OUTPUT_ID}
    )
    assert all(not t3.result.artifacts[c].evidence for c in checks)

    # Q4 add_dimension(pricing): pricing nodes added. The checks re-run (their inputs gained
    # the pricing branch) but still find nothing, so the gate decision is unchanged and the
    # RPO retrieval branches are reused rather than re-run.
    assert [o.kind for o in t4.compiled.ops] == [RevisionKind.ADD_DIMENSION]
    assert t4.result.reused == aws_dr | gcp_dr | _branch("aws", "rpo") | _branch("gcp", "rpo")
    assert t4.result.executed == (
        checks
        | _branch("aws", "pricing")
        | _branch("gcp", "pricing")
        | {"aws__evidence", "gcp__evidence", OUTPUT_ID}
    )
    assert t4.removed == frozenset()
    assert {"aws-pricing", "gcp-pricing"} <= {e.source_id for e in t4.result.output.evidence}


def test_worked_example_final_plan_matches_full_execution(
    index: BM25Index, dense_index: DenseIndex
) -> None:
    seq = RECORDED["sequence"]
    inc = _session(index, dense_index, ScriptedLLM([s["output"] for s in seq]))
    full = _session(index, dense_index, ScriptedLLM([seq[-1]["output"]]))

    async def go() -> tuple[Turn, Turn]:
        for s in seq[:-1]:
            await inc.ask(s["text"])
        # A fresh session needs the self-contained form of the final question.
        standalone = "Compare AWS and GCP on disaster recovery and pricing with an RPO of zero"
        return await inc.ask(seq[-1]["text"]), await full.ask(standalone)

    t_inc, t_full = run(go())
    assert t_full.result.reused == set()
    assert t_inc.plan == t_full.plan
    for n in t_inc.plan.nodes:
        assert content(t_inc.result.artifacts[n.id]) == content(t_full.result.artifacts[n.id])


def test_compile_failure_executes_nothing(index: BM25Index, dense_index: DenseIndex) -> None:
    session = _session(index, dense_index, ScriptedLLM([LLMError("down"), LLMError("down")]))
    with pytest.raises(CompileError):
        run(session.ask("Compare AWS and Azure"))
    assert session.turns == []
    assert run(session.executor.store.count_artifacts()) == 0
