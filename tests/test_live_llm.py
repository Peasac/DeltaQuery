"""Live checks against the local Ollama server: `uv run pytest -m live`.

Models: DQ_LIVE_MODELS (comma-separated), default qwen3.5:4b. These are single runs at
temperature 0, i.e. smoke checks, not accuracy measurements.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from deltaquery.compiler.compiler import IntentCompiler
from deltaquery.compiler.intent import Intent
from deltaquery.llm.ollama import OllamaLLM

pytestmark = pytest.mark.live
MODELS = os.environ.get("DQ_LIVE_MODELS", "qwen3.5:4b").split(",")
ENTITIES = {"aws", "azure", "gcp"}
SEQ = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded_intents.json").read_text(encoding="utf-8")
)["sequence"]


@pytest.mark.parametrize("model", MODELS)
def test_live_compiler_follows_worked_example(model: str) -> None:
    async def go() -> list[Intent]:
        llm = await OllamaLLM.connect(model)
        compiler = IntentCompiler(llm, ENTITIES, timeout_s=180)
        prior: Intent | None = None
        out = []
        for step in SEQ:
            prior = (await compiler.compile(step["text"], prior)).intent
            out.append(prior)
        return out

    got = asyncio.run(go())
    expected = [Intent.model_validate(s["output"], context={"entities": ENTITIES}) for s in SEQ]
    for step, g, e in zip(SEQ, got, expected, strict=True):
        assert g == e, f"{model} on {step['text']!r}"


@pytest.mark.parametrize("model", MODELS)
def test_live_sdk_answers_with_valid_citations(model: str) -> None:
    from deltaquery.evidence.access import AccessPolicy
    from deltaquery.evidence.corpus import Corpus
    from deltaquery.sdk import Runtime

    corpus = Corpus.from_json(Path(__file__).parent / "fixtures" / "corpus.json")
    policy = AccessPolicy()
    policy.grant("alice", "demo")

    async def go() -> tuple[object, object]:
        llm = await OllamaLLM.connect(model)
        runtime = await Runtime.create(corpus=corpus, llm=llm, policy=policy, entities=ENTITIES)
        session = runtime.session(scope="demo", principal="alice")
        first = await session.ask(SEQ[0]["text"])
        nxt = await session.ask(SEQ[1]["text"])
        await runtime.close()
        return first, nxt

    first, nxt = asyncio.run(go())
    assert nxt.trace.reused_nodes  # type: ignore[attr-defined]
    for turn in (first, nxt):
        ans = turn.answer  # type: ignore[attr-defined]
        print(model, turn.text, "->", ans.reason if ans.abstained else ans.text)  # type: ignore[attr-defined]
        # Citations are validated by construction; here we only require that it answered.
        assert not ans.abstained, ans.reason
        assert all(s.citations for s in ans.sentences)
