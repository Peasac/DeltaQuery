"""Minimal revision session: compile -> plan -> execute, carrying the prior intent forward.

Reuse across turns comes entirely from content-addressed signatures in the shared store; the
session only remembers the prior intent (for the compiler) and prior plan (for the trace).
Epoch/cancellation of superseded turns is Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from deltaquery.compiler.compiler import CompileResult, IntentCompiler
from deltaquery.compiler.intent import Intent
from deltaquery.compiler.planner import Planner
from deltaquery.runtime.executor import ExecutionResult, Executor
from deltaquery.runtime.models import Plan


@dataclass(frozen=True)
class Turn:
    text: str
    compiled: CompileResult
    plan: Plan
    result: ExecutionResult
    removed: frozenset[str]  # node ids in the previous plan but not in this one

    @property
    def intent(self) -> Intent:
        return self.compiled.intent


class Session:
    def __init__(self, compiler: IntentCompiler, planner: Planner, executor: Executor) -> None:
        self.compiler = compiler
        self.planner = planner
        self.executor = executor
        self.turns: list[Turn] = []

    async def ask(self, text: str) -> Turn:
        prior = self.turns[-1] if self.turns else None
        compiled = await self.compiler.compile(text, prior.intent if prior else None)
        plan = self.planner.build(compiled.intent)
        result = await self.executor.execute(plan)
        prev_ids = {n.id for n in prior.plan.nodes} if prior else set()
        turn = Turn(
            text=text,
            compiled=compiled,
            plan=plan,
            result=result,
            removed=frozenset(prev_ids - {n.id for n in plan.nodes}),
        )
        self.turns.append(turn)
        return turn
