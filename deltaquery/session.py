"""Revision session: compile -> plan -> execute -> (optionally) answer.

Epochs: every `ask` starts a new epoch and cancels the in-flight one. A turn commits only if
its epoch is still current when it finishes, so a late result from a superseded request can
never become session state, even if its work ignored cancellation. Artifacts that superseded
work already stored are content-addressed and valid; they are not session state.

`expected_version` (optimistic concurrency): if given, it must equal the number of committed
turns, otherwise the ask is rejected before doing any work.

Revisions are compiled against the last *committed* intent.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from deltaquery.compiler.compiler import CompileResult, IntentCompiler
from deltaquery.compiler.intent import Intent, render_question
from deltaquery.compiler.planner import Planner
from deltaquery.runtime.answer import Answer, AnswerGenerator
from deltaquery.runtime.executor import ExecutionResult, Executor, NodeStatus
from deltaquery.runtime.models import Plan


class SupersededError(RuntimeError):
    """A newer ask started before this one committed; its result was discarded."""


class VersionConflict(RuntimeError):
    """`expected_version` did not match the session's committed version."""


@dataclass(frozen=True)
class TurnTrace:
    version: int
    epoch: int
    executed_nodes: tuple[str, ...]
    reused_nodes: tuple[str, ...]
    skipped_nodes: tuple[str, ...]  # gated off, e.g. existing evidence was sufficient
    removed_nodes: tuple[str, ...]
    reuse_rejections: tuple[tuple[str, str], ...]  # (node_id, reason) for signature hits refused
    compile_s: float
    retrieval_s: float
    generation_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epoch": self.epoch,
            "executed_nodes": list(self.executed_nodes),
            "reused_nodes": list(self.reused_nodes),
            "skipped_nodes": list(self.skipped_nodes),
            "removed_nodes": list(self.removed_nodes),
            "reuse_rejections": [list(r) for r in self.reuse_rejections],
            "timing_s": {
                "compile": self.compile_s,
                "retrieval": self.retrieval_s,
                "generation": self.generation_s,
            },
        }


@dataclass(frozen=True)
class Turn:
    text: str
    compiled: CompileResult
    plan: Plan
    result: ExecutionResult
    answer: Answer | None
    trace: TurnTrace

    @property
    def intent(self) -> Intent:
        return self.compiled.intent

    @property
    def removed(self) -> frozenset[str]:
        return frozenset(self.trace.removed_nodes)


class Session:
    def __init__(
        self,
        compiler: IntentCompiler,
        planner: Planner,
        executor: Executor,
        *,
        principal: str,
        generator: AnswerGenerator | None = None,
    ) -> None:
        # Fail closed at creation too, not only at execution time.
        executor.policy.check(principal, planner.c.access_scope)
        self.compiler = compiler
        self.planner = planner
        self.executor = executor
        self.principal = principal
        self.generator = generator
        self.turns: list[Turn] = []
        self._epoch = 0
        self._inflight: asyncio.Task[Turn] | None = None

    @property
    def version(self) -> int:
        return len(self.turns)

    async def ask(self, text: str, *, expected_version: int | None = None) -> Turn:
        if expected_version is not None and expected_version != self.version:
            raise VersionConflict(f"expected version {expected_version}, at {self.version}")
        self._epoch += 1
        epoch = self._epoch
        if self._inflight is not None and not self._inflight.done():
            self._inflight.cancel()
        prior = self.turns[-1] if self.turns else None
        task = asyncio.create_task(self._run(text, prior, epoch))
        self._inflight = task
        try:
            turn = await task
        except asyncio.CancelledError:
            if epoch != self._epoch:
                raise SupersededError(f"epoch {epoch} superseded by {self._epoch}") from None
            raise
        if epoch != self._epoch:  # finished, but too late: never commit
            raise SupersededError(f"epoch {epoch} superseded by {self._epoch}")
        self.turns.append(turn)
        return turn

    async def _run(self, text: str, prior: Turn | None, epoch: int) -> Turn:
        t0 = time.perf_counter()
        compiled = await self.compiler.compile(text, prior.intent if prior else None)
        t1 = time.perf_counter()
        plan = self.planner.build(compiled.intent)
        result = await self.executor.execute(plan, principal=self.principal)
        t2 = time.perf_counter()
        answer = None
        if self.generator is not None:
            question = f"{render_question(compiled.intent)} (Latest message: {text})"
            answer = await self.generator.generate(question, result.output.evidence)
        t3 = time.perf_counter()
        prev_ids = {n.id for n in prior.plan.nodes} if prior else set()
        ids = {n.id for n in plan.nodes}
        trace = TurnTrace(
            version=self.version + 1,
            epoch=epoch,
            executed_nodes=tuple(sorted(result.ids_with(NodeStatus.EXECUTED))),
            reused_nodes=tuple(sorted(result.ids_with(NodeStatus.REUSED))),
            skipped_nodes=tuple(sorted(result.ids_with(NodeStatus.SKIPPED))),
            removed_nodes=tuple(sorted(prev_ids - ids)),
            reuse_rejections=tuple(
                (t.node_id, t.reuse_rejected) for t in result.trace if t.reuse_rejected
            ),
            compile_s=t1 - t0,
            retrieval_s=t2 - t1,
            generation_s=t3 - t2,
        )
        return Turn(text, compiled, plan, result, answer, trace)
