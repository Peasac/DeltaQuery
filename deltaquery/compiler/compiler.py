"""Structured-output intent compiler.

The LLM sees only the user's text and the prior intent (never retrieved text), and must return
a complete intent under a JSON schema with closed vocabularies. Output is re-validated here;
on invalid output we retry once, then fail with CompileError (no guessing).

With grounding on (default), each change the model proposes relative to the prior intent is
approved only if the message names it (compiler/grounding.py), `task` is derived from the entity
count, and a message that names no slot at all carries the prior intent forward without calling
the model. A fresh question that names no provider fails with CompileError.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from deltaquery.compiler.diff import RevisionOp, diff_intents
from deltaquery.compiler.grounding import Mentions, ground, mentions, task_for
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
    attempts: int  # 0 when the prior intent was carried forward without calling the model
    grounding_notes: tuple[str, ...] = ()


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
        grounding: bool = True,
    ) -> None:
        self.llm = llm
        self.entities = frozenset(entities)
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.grounding = grounding
        self._schema = intent_json_schema(self.entities)

    def _result(
        self, prior: Intent | None, intent: Intent, attempts: int, notes: tuple[str, ...] = ()
    ) -> CompileResult:
        return CompileResult(
            intent=intent,
            ops=diff_intents(prior, intent),
            model_version=self.llm.model_version,
            prompt_version=PROMPT_VERSION,
            attempts=attempts,
            grounding_notes=notes,
        )

    def _apply_grounding(
        self, text: str, prior: Intent | None, proposed: Intent, m: Mentions
    ) -> tuple[Intent, tuple[str, ...]]:
        g = ground(text, prior, proposed, m)
        if not g.entities:
            raise CompileError(f"no provider named in a new question: {text!r}"[:300])
        intent = Intent.model_validate(
            {
                "task": task_for(g.entities),
                "entities": list(g.entities),
                "dimensions": list(g.dimensions),
                "constraints": [c.model_dump(mode="json") for c in g.constraints],
            },
            context={"entities": self.entities},
        )
        return intent, g.notes

    async def compile(self, text: str, prior: Intent | None = None) -> CompileResult:
        m = mentions(text, self.entities) if self.grounding else None
        if m is not None and m.empty:
            if prior is None:
                raise CompileError(f"message names no provider, topic or attribute: {text!r}"[:300])
            # Nothing to change: carry the prior intent forward, no model call.
            return self._result(prior, prior, 0, ("carried forward: no slot named",))
        user = build_user_prompt(text, prior, self.entities)
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = await self.llm.complete_json(
                    system=SYSTEM_PROMPT, user=user, schema=self._schema, timeout_s=self.timeout_s
                )
                if m is not None and isinstance(raw.get("entities"), list):
                    raw = {**raw, "task": task_for(raw["entities"])}  # task follows entities
                intent = Intent.model_validate(raw, context={"entities": self.entities})
            except (LLMError, ValidationError, TimeoutError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}"[:500])
                continue
            if m is None:
                return self._result(prior, intent, attempt)
            grounded, notes = self._apply_grounding(text, prior, intent, m)
            return self._result(prior, grounded, attempt, notes)
        raise CompileError("; ".join(errors))
