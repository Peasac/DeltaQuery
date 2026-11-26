"""Grounding of LLM intent proposals: each fixture reproduces a failure seen in the e2e run
(project.md Phase 5) with a scripted model output."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deltaquery.compiler.compiler import CompileError, IntentCompiler
from deltaquery.compiler.grounding import mentions
from deltaquery.compiler.intent import Dimension, Intent

VOCAB = {"ardor", "arthas", "delmora", "delwick", "jornix", "pelvane", "kelmora"}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class ScriptedLLM:
    model_version = "scripted/test@0"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls = 0

    async def complete_json(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return self.outputs.pop(0)


def intent(ents: list[str], dims: list[str], cons: list[dict[str, Any]] | None = None) -> Intent:
    return Intent.model_validate(
        {"task": "compare" if len(ents) > 1 else "describe", "entities": ents,
         "dimensions": dims, "constraints": cons or []},
        context={"entities": VOCAB},
    )  # fmt: skip


def out(
    task: str, ents: list[str], dims: list[str], cons: list[Any] | None = None
) -> dict[str, Any]:
    return {"task": task, "entities": ents, "dimensions": dims, "constraints": cons or []}


RPO0 = {"attribute": "rpo", "op": "eq", "value_seconds": 0}
PRIOR = intent(["jornix", "pelvane"], ["pricing"])


def compile_(text: str, prior: Intent | None, proposal: dict[str, Any], **kw: Any) -> Any:
    llm = ScriptedLLM([proposal])
    r = run(IntentCompiler(llm, VOCAB, **kw).compile(text, prior))
    return r, llm


def test_neutral_follow_up_carries_prior_without_calling_the_model() -> None:
    for text in ("Check again with the latest documents", "Show that again"):
        r, llm = compile_(text, PRIOR, out("describe", ["ardor", "arthas"], ["backup"]))
        assert r.intent == PRIOR and llm.calls == 0 and r.attempts == 0


def test_enum_order_hallucination_is_rejected() -> None:
    # Observed: providers replaced by the first entries of the sorted enum.
    proposal = out("compare", ["ardor", "arthas", "delmora", "delwick"], ["pricing"])
    r, _ = compile_("Also compare pricing", PRIOR, proposal)
    assert r.intent.entities == PRIOR.entities
    assert any("dropped unnamed entity ardor" in n for n in r.grounding_notes)


def test_task_flip_is_normalized() -> None:
    r, _ = compile_(
        "Also compare pricing", PRIOR, out("describe", ["jornix", "pelvane"], ["pricing"])
    )
    assert r.intent.task.value == "compare"


def test_constraint_does_not_smuggle_in_a_topic() -> None:
    # Observed: "RPO of zero" also added disaster_recovery.
    proposal = out("compare", ["jornix", "pelvane"], ["disaster_recovery", "pricing"], [RPO0])
    r, _ = compile_("Only consider options with an RPO of zero", PRIOR, proposal)
    assert r.intent.dimensions == (Dimension.PRICING,)
    assert [c.value_seconds for c in r.intent.constraints] == [0]


def test_unrequested_removal_is_restored() -> None:
    r, _ = compile_(
        "Also include Kelmora", PRIOR, out("compare", ["kelmora", "pelvane"], ["pricing"])
    )
    assert r.intent.entities == ("jornix", "kelmora", "pelvane")


def test_requested_changes_pass_through() -> None:
    r, _ = compile_(
        "Replace Jornix with Kelmora", PRIOR, out("compare", ["kelmora", "pelvane"], ["pricing"])
    )
    assert r.intent.entities == ("kelmora", "pelvane")
    prior = intent(["jornix", "pelvane"], ["pricing", "backup"], [RPO0])
    r, _ = compile_(
        "Ignore backup now", prior, out("compare", ["jornix", "pelvane"], ["pricing"], [RPO0])
    )
    assert r.intent.dimensions == (Dimension.PRICING,)
    r, _ = compile_(
        "Remove the RPO requirement",
        prior,
        out("compare", ["jornix", "pelvane"], ["backup", "pricing"]),
    )
    assert r.intent.constraints == ()


def test_constraint_value_must_be_stated() -> None:
    wrong = {"attribute": "rto", "op": "le", "value_seconds": 24}  # observed: invented value
    r, _ = compile_(
        "Only consider options with an RTO of at most 900 seconds",
        PRIOR,
        out("compare", ["jornix", "pelvane"], ["pricing"], [wrong]),
    )
    assert r.intent.constraints == ()  # 24 s is not what the user said
    assert any("ungrounded constraint rto" in n for n in r.grounding_notes)


def test_unrelated_question_replaces_the_prior() -> None:
    r, _ = compile_(
        "Compare Ardor and Arthas on regions",
        PRIOR,
        out("compare", ["ardor", "arthas"], ["regions"]),
    )
    assert r.intent == intent(["ardor", "arthas"], ["regions"])


def test_fresh_question_includes_every_named_provider() -> None:
    # e2e failure: the model listed two of the three providers the question named.
    r, _ = compile_(
        "Compare Ardor, Arthas and Delwick on backup",
        None,
        out("compare", ["ardor", "arthas"], ["backup"]),
    )
    assert r.intent == intent(["ardor", "arthas", "delwick"], ["backup"])
    assert any("added named entity delwick" in n for n in r.grounding_notes)


def test_fresh_question_without_a_provider_fails_closed() -> None:
    with pytest.raises(CompileError):
        compile_("Also compare pricing", None, out("compare", ["ardor", "arthas"], ["pricing"]))
    with pytest.raises(CompileError):
        compile_("hello", None, out("compare", ["ardor", "arthas"], ["pricing"]))


def test_grounding_off_passes_model_output_through() -> None:
    proposal = out("compare", ["ardor", "arthas", "delmora", "delwick"], ["pricing"])
    r, _ = compile_("Also compare pricing", PRIOR, proposal, grounding=False)
    assert r.intent.entities == ("ardor", "arthas", "delmora", "delwick")


def test_mentions_lexicon() -> None:
    m = mentions("DR and costs for Delmora; recovery time under 1 minute", VOCAB)
    assert m.entities == {"delmora"}
    assert m.dimensions == {Dimension.DISASTER_RECOVERY, Dimension.PRICING}
    assert {a.value for a in m.attributes} == {"rto"}
