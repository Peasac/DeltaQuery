"""Async DAG executor with signature-based reuse.

For each topological layer, independent nodes are resolved concurrently:
    compute signature from (node, operator version, input artifact hashes)
    -> look up artifact -> validate -> reuse, or execute (with timeout) and store.

Access: the principal must hold every scope in the plan (checked before anything runs); the
scope's access-policy version is part of each signature; revoked sources are excluded from
retrieval, re-checked on every reuse, and asserted absent from every new artifact.

`reuse=False` is full re-execution with the same planner. It still writes to the store, and the
store raises DeterminismViolation if a signature ever yields different content.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from deltaquery.canonical import hash_canonical
from deltaquery.evidence.access import AccessDenied, AccessPolicy
from deltaquery.evidence.models import Artifact, Evidence, compute_artifact_hash
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import BoundVersions, ExecutionContext, Operator
from deltaquery.runtime.models import FreshnessPolicy, Node, Plan, PlanValidationError
from deltaquery.runtime.signature import node_signature
from deltaquery.runtime.validate import ReuseContext, reuse_rejection_reason


class NodeStatus(StrEnum):
    EXECUTED = "executed"
    REUSED = "reused"
    SKIPPED = "skipped"  # gated off (e.g. existing evidence was sufficient)


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


OperatorKey = tuple[str, str | None, str | None, str | None]


def operator_key(name: str, v: BoundVersions) -> OperatorKey:
    return (name, v.source_snapshot, v.index_version, v.model_version)


def access_fingerprint(scope: str, revoked: frozenset[str]) -> str:
    """Same formula as AccessPolicy.version, so the coarse mode reproduces it exactly."""
    return hash_canonical("access", {"scope": scope, "revoked": sorted(revoked)})


@dataclass(frozen=True)
class _ScopeView:
    version: str
    revoked: frozenset[str]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Executor:
    def __init__(
        self,
        store: EvidenceStore,
        operators: Iterable[Operator[Any]],
        policy: AccessPolicy,
        *,
        node_timeout_s: float = 30.0,
        clock: Callable[[], datetime] = _utcnow,
        precise_access: bool = True,
    ) -> None:
        # Versioned registry: the same operator name may be bound to several snapshots
        # (e.g. old and new index after a source update). Nodes are routed by their
        # declared versions; there is no "closest match".
        self._registry: dict[OperatorKey, Operator[Any]] = {}
        for op in operators:
            key = operator_key(op.name, op.bound_versions)
            if key in self._registry:
                raise ValueError(f"duplicate operator registration: {key}")
            self._registry[key] = op
        self._names = {k[0] for k in self._registry}
        self.store = store
        self.policy = policy
        # precise_access: a node's signature carries only the revocations its own result can
        # depend on (Operator.access_dependencies). False = the whole scope's policy version
        # (coarse; any revocation invalidates every artifact in the scope).
        self.precise_access = precise_access
        self.node_timeout_s = node_timeout_s
        self.clock = clock

    def operator_for(self, node: Node) -> Operator[Any]:
        if node.operator not in self._names:
            raise PlanValidationError(f"node {node.id!r}: unknown operator {node.operator!r}")
        key = (node.operator, node.source_snapshot, node.index_version, node.model_version)
        op = self._registry.get(key)
        if op is None:
            raise PlanValidationError(
                f"node {node.id!r}: declared versions do not match any registered "
                f"{node.operator!r} operator (snapshot/index/model not available)"
            )
        return op

    def check_plan(self, plan: Plan) -> None:
        """Static checks against the operator registry. Raises PlanValidationError."""
        for n in plan.nodes:
            op = self.operator_for(n)
            if n.output_contract != op.output_contract:
                raise PlanValidationError(f"node {n.id!r}: output contract mismatch")
            k = len(n.input_ids)
            if k < op.min_inputs or (op.max_inputs is not None and k > op.max_inputs):
                raise PlanValidationError(f"node {n.id!r}: bad input arity {k}")
            try:
                op.parse_params(n.canonical_parameters)
            except ValueError as exc:
                raise PlanValidationError(f"node {n.id!r}: {exc}") from exc

    async def execute(self, plan: Plan, *, principal: str, reuse: bool = True) -> ExecutionResult:
        """Execute `plan` on behalf of `principal`. Fails closed (AccessDenied) before any
        reuse or execution if the principal may not read every scope the plan touches."""
        self.check_plan(plan)
        scopes = {n.access_scope for n in plan.nodes}
        for scope in sorted(scopes):
            self.policy.check(principal, scope)
        # One consistent view of the policy for the whole plan.
        views = {
            s: _ScopeView(self.policy.version(s), self.policy.revoked_sources(s)) for s in scopes
        }
        artifacts: dict[str, Artifact] = {}
        traces: list[NodeTrace] = []
        for layer in plan.layers():
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._resolve(n, artifacts, views[n.access_scope], principal, reuse)
                    )
                    for n in layer
                ]
            for node, task in zip(layer, tasks, strict=True):
                artifact, trace = task.result()
                artifacts[node.id] = artifact
                traces.append(trace)
        return ExecutionResult(plan=plan, artifacts=artifacts, trace=tuple(traces))

    @staticmethod
    def _artifact(
        node: Node,
        op: Operator[Any],
        sig: str,
        access_fp: str,
        inputs: Sequence[Artifact],
        evidence: tuple[Evidence, ...],
        now: datetime,
    ) -> Artifact:
        return Artifact(
            signature=sig,
            operator=op.name,
            operator_version=op.version,
            output_contract=op.output_contract,
            access_scope=node.access_scope,
            access_policy_version=access_fp,
            source_snapshot=node.source_snapshot,
            index_version=node.index_version,
            model_version=node.model_version,
            input_signatures=tuple(a.signature for a in inputs),
            evidence=evidence,
            artifact_hash=compute_artifact_hash(op.output_contract, evidence),
            created_at=now,
        )

    async def _resolve(
        self,
        node: Node,
        done: Mapping[str, Artifact],
        view: _ScopeView,
        principal: str,
        reuse: bool,
    ) -> tuple[Artifact, NodeTrace]:
        op = self.operator_for(node)
        inputs = [done[i] for i in node.input_ids]
        params = op.parse_params(node.canonical_parameters)
        deps = (
            op.access_dependencies(params, node.access_scope, view.revoked)
            if self.precise_access
            else view.revoked
        )
        access_fp = access_fingerprint(node.access_scope, deps)
        gate: str | None = None
        if node.skip_if_nonempty is not None:
            gate = "skip" if done[node.skip_if_nonempty].evidence else "run"
        sig = node_signature(
            node, op.version, [a.artifact_hash for a in inputs], access_fp, gate=gate
        )
        if gate == "skip":
            skipped = self._artifact(node, op, sig, access_fp, inputs, (), self.clock())
            return skipped, NodeTrace(node.id, node.operator, sig, NodeStatus.SKIPPED)

        rejected: str | None = None
        if reuse:
            try:
                cached = await self.store.get_artifact(sig)
            except ValueError:  # stored manifest fails its integrity check
                cached, rejected = None, "corrupt"
            if cached is not None:
                now = self.clock()
                last_verified = (
                    await self.store.last_verified_at(sig)
                    if node.freshness_policy is FreshnessPolicy.TTL
                    else None
                )
                rctx = ReuseContext(
                    signature=sig,
                    inputs=inputs,
                    access_policy_version=access_fp,
                    revoked_sources=view.revoked,
                    principal_may_read=self.policy.can_read(principal, node.access_scope),
                    now=now,
                    last_verified_at=last_verified,
                )
                rejected = reuse_rejection_reason(node, op, cached, rctx)
                if rejected is None:
                    await self.store.touch(sig, now)
                    return cached, NodeTrace(node.id, node.operator, sig, NodeStatus.REUSED)

        now = self.clock()
        ctx = ExecutionContext(
            signature=sig,
            access_scope=node.access_scope,
            now=now,
            excluded_sources=view.revoked,
        )
        evidence = tuple(
            await asyncio.wait_for(op.run(params, inputs, ctx), timeout=self.node_timeout_s)
        )
        leaked = {e.source_id for e in evidence} & view.revoked
        if leaked:  # operator bug: never store or return revoked content
            raise AccessDenied(f"node {node.id!r} returned revoked sources {sorted(leaked)}")
        artifact = self._artifact(node, op, sig, access_fp, inputs, evidence, now)
        await self.store.put_artifact(artifact)
        return artifact, NodeTrace(node.id, node.operator, sig, NodeStatus.EXECUTED, rejected)
