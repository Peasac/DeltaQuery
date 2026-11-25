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
