"""Async DAG executor with signature-based reuse.

For each topological layer, independent nodes are resolved concurrently:
    compute signature from (node, operator version, input artifact hashes)
    -> look up artifact -> validate -> reuse, or execute (with timeout) and store.

`reuse=False` is full re-execution with the same planner. It still writes to the store, and the
store raises DeterminismViolation if a signature ever yields different content.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from deltaquery.evidence.models import Artifact, compute_artifact_hash
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext, Operator
from deltaquery.runtime.models import Node, Plan, PlanValidationError
from deltaquery.runtime.signature import node_signature
from deltaquery.runtime.validate import reuse_rejection_reason


class NodeStatus(StrEnum):
    EXECUTED = "executed"
    REUSED = "reused"


@dataclass(frozen=True)
class NodeTrace:
    node_id: str
    operator: str
    signature: str
    status: NodeStatus
    reuse_rejected: str | None = None  # why a signature hit was not reused, if it wasn't


@dataclass(frozen=True)
class ExecutionResult:
    plan: Plan
    artifacts: Mapping[str, Artifact]  # by plan-local node id
    trace: tuple[NodeTrace, ...]

    @property
    def output(self) -> Artifact:
        return self.artifacts[self.plan.output_id]

    def ids_with(self, status: NodeStatus) -> set[str]:
        return {t.node_id for t in self.trace if t.status is status}

    @property
    def executed(self) -> set[str]:
        return self.ids_with(NodeStatus.EXECUTED)

    @property
    def reused(self) -> set[str]:
        return self.ids_with(NodeStatus.REUSED)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Executor:
    def __init__(
        self,
        store: EvidenceStore,
        operators: Mapping[str, Operator[Any]],
        *,
        node_timeout_s: float = 30.0,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        for name, op in operators.items():
            if name != op.name:
                raise ValueError(f"operator registered as {name!r} but named {op.name!r}")
        self.store = store
        self.operators = dict(operators)
        self.node_timeout_s = node_timeout_s
        self.clock = clock

    def check_plan(self, plan: Plan) -> None:
        """Static checks against the operator registry. Raises PlanValidationError."""
        for n in plan.nodes:
            op = self.operators.get(n.operator)
            if op is None:
                raise PlanValidationError(f"node {n.id!r}: unknown operator {n.operator!r}")
            if n.output_contract != op.output_contract:
                raise PlanValidationError(f"node {n.id!r}: output contract mismatch")
            k = len(n.input_ids)
            if k < op.min_inputs or (op.max_inputs is not None and k > op.max_inputs):
                raise PlanValidationError(f"node {n.id!r}: bad input arity {k}")
            bound = op.bound_versions
            declared = (n.source_snapshot, n.index_version, n.model_version)
            actual = (bound.source_snapshot, bound.index_version, bound.model_version)
            if declared != actual:
                raise PlanValidationError(
                    f"node {n.id!r}: declared versions do not match the bound operator"
                )
            try:
                op.parse_params(n.canonical_parameters)
            except ValueError as exc:
                raise PlanValidationError(f"node {n.id!r}: {exc}") from exc

    async def execute(self, plan: Plan, *, reuse: bool = True) -> ExecutionResult:
        self.check_plan(plan)
        artifacts: dict[str, Artifact] = {}
        traces: list[NodeTrace] = []
        for layer in plan.layers():
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._resolve(n, artifacts, reuse)) for n in layer]
            for node, task in zip(layer, tasks, strict=True):
                artifact, trace = task.result()
                artifacts[node.id] = artifact
                traces.append(trace)
        return ExecutionResult(plan=plan, artifacts=artifacts, trace=tuple(traces))

    async def _resolve(
        self, node: Node, done: Mapping[str, Artifact], reuse: bool
    ) -> tuple[Artifact, NodeTrace]:
        op = self.operators[node.operator]
        inputs = [done[i] for i in node.input_ids]
        sig = node_signature(node, op.version, [a.artifact_hash for a in inputs])

        rejected: str | None = None
        if reuse:
            cached = await self.store.get_artifact(sig)
            if cached is not None:
                rejected = reuse_rejection_reason(node, op, sig, inputs, cached)
                if rejected is None:
                    return cached, NodeTrace(node.id, node.operator, sig, NodeStatus.REUSED)

        params = op.parse_params(node.canonical_parameters)
        now = self.clock()
        ctx = ExecutionContext(signature=sig, access_scope=node.access_scope, now=now)
        evidence = tuple(
            await asyncio.wait_for(op.run(params, inputs, ctx), timeout=self.node_timeout_s)
        )
        artifact = Artifact(
            signature=sig,
            operator=op.name,
            operator_version=op.version,
            output_contract=op.output_contract,
            access_scope=node.access_scope,
            source_snapshot=node.source_snapshot,
            index_version=node.index_version,
            model_version=node.model_version,
            input_signatures=tuple(a.signature for a in inputs),
            evidence=evidence,
            artifact_hash=compute_artifact_hash(op.output_contract, evidence),
            created_at=now,
        )
        await self.store.put_artifact(artifact)
        return artifact, NodeTrace(node.id, node.operator, sig, NodeStatus.EXECUTED, rejected)
