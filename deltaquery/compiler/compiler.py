"""Structured-output intent compiler.

The LLM sees only the user's text and the prior intent (never retrieved text), and must return
a complete intent under a JSON schema with closed vocabularies. Output is re-validated here;
on invalid output we retry once, then fail with CompileError (no guessing).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from deltaquery.compiler.diff import RevisionOp, diff_intents
from deltaquery.compiler.intent import (
    Attribute,
    Comparison,
    Dimension,
    Intent,
    Task,
    intent_json_schema,
)
from deltaquery.llm.base import LLMError, StructuredLLM

PROMPT_VERSION = "intent-v1"

SYSTEM_PROMPT = f"""You turn a user's question about cloud providers into a JSON retrieval intent.

Fields:
- task: "{Task.COMPARE.value}" when two or more providers are compared, else "{Task.DESCRIBE.value}".
- entities: providers mentioned, from the allowed list only.
- dimensions: topics asked about, from: {", ".join(d.value for d in Dimension)}.
  "DR" means disaster_recovery.
- constraints: numeric requirements, each {{"attribute": one of {[a.value for a in Attribute]},
  "op": one of {[c.value for c in Comparison]} ("eq" for equal, "le" for at most),
  "value_seconds": integer seconds}}. "RPO of zero" -> {{"attribute":"rpo","op":"eq","value_seconds":0}}.

If a PREVIOUS INTENT is given, the new message revises it. Output the COMPLETE updated intent:
apply only the change the user asks for and keep every other field exactly as before.
"Replace X with Y" removes X and adds Y. "Also consider/add Z" adds Z.
If the new message is an unrelated question, ignore the previous intent.
Output JSON only."""


class CompileError(ValueError):
    """The compiler could not produce a valid intent. Callers must not guess one."""


@dataclass(frozen=True)
class CompileResult:
    intent: Intent
    ops: tuple[RevisionOp, ...]
    model_version: str
    prompt_version: str
    attempts: int


def build_user_prompt(text: str, prior: Intent | None, entities: Iterable[str]) -> str:
    prev = "none" if prior is None else json.dumps(prior.model_dump(mode="json"), sort_keys=True)
    return (
        f"Allowed entities: {', '.join(sorted(entities))}\n"
        f"PREVIOUS INTENT: {prev}\n"
        f"NEW MESSAGE: {text}"
    )


class IntentCompiler:
    def __init__(
        self,
        llm: StructuredLLM,
        entities: Iterable[str],
        *,
        timeout_s: float = 60.0,
        max_attempts: int = 2,
    ) -> None:
        self.llm = llm
        self.entities = frozenset(entities)
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self._schema = intent_json_schema(self.entities)

    async def compile(self, text: str, prior: Intent | None = None) -> CompileResult:
        user = build_user_prompt(text, prior, self.entities)
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = await self.llm.complete_json(
                    system=SYSTEM_PROMPT, user=user, schema=self._schema, timeout_s=self.timeout_s
                )
                intent = Intent.model_validate(raw, context={"entities": self.entities})
            except (LLMError, ValidationError, TimeoutError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}"[:500])
                continue
            return CompileResult(
                intent=intent,
                ops=diff_intents(prior, intent),
                model_version=self.llm.model_version,
                prompt_version=PROMPT_VERSION,
                attempts=attempt,
            )
        raise CompileError("; ".join(errors))
