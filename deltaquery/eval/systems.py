"""Systems compared on DeltaBench (oracle-intent track).

All systems except `semantic` and `conv_hybrid` run the *same* plans from the same planner over
the same operators, so differences come only from what they reuse.

- deltaquery       the runtime (signatures + validator + access policy), one store per run
- full             same planner, everything re-executed every turn
- exact_naive      per-operator cache keyed on (operator, params, input content); unaware of
                   snapshot/index/model versions and of the access policy
- exact_versioned  exact_naive + snapshot/index/model versions in the key (a competent exact
                   operator cache). Unaware of the access policy. The decisive baseline.
- semantic         answer-level cache: if the rendered question's embedding is within a cosine
                   threshold of a cached question, return that turn's evidence; else run fully
- conv_hybrid      one rewritten query per turn (bm25 + dense over everything, rrf); no
                   planner, no cache
Ablations of deltaquery (same cache machinery as the exact baselines, keys toggled):
- abl_no_deps      key without input content (no dependency propagation)
- abl_no_versions  = exact_naive + policy version (versions dropped)
- abl_no_policy    = exact_versioned (policy dropped); listed for the ablation table
- dq_coarse_policy deltaquery with whole-scope policy versions in signatures (pre-lever-1)
- dq_no_memcache   deltaquery without the in-process artifact cache (pre-lever-2)

Every cache-based system executes fresh work under the *current* policy (revoked sources are
excluded at retrieval). Stale results can therefore only come from reuse.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from deltaquery.canonical import hash_canonical
from deltaquery.compiler.intent import Intent, render_question
from deltaquery.compiler.planner import Planner, PlannerConfig
from deltaquery.evidence.access import AccessPolicy
from deltaquery.evidence.corpus import Corpus, PassageSet
from deltaquery.evidence.models import Artifact, Evidence, compute_artifact_hash
from deltaquery.evidence.store import EvidenceStore
from deltaquery.operators.base import ExecutionContext, Operator
from deltaquery.operators.bm25 import BM25Index, BM25Operator, BM25Params
from deltaquery.operators.dense import DenseIndex, DenseOperator, DenseParams
from deltaquery.operators.embedding import HashingEmbedder, l2_normalize
from deltaquery.operators.fusion import RRFOperator, RRFParams
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.executor import Executor, NodeStatus
from deltaquery.runtime.models import Node, Plan

PRINCIPAL = "bench"
RETRIEVERS = frozenset({"bm25", "dense"})


@dataclass
class Snapshot:
    corpus: Corpus
    passages: PassageSet
    bm25: BM25Index
    dense: DenseIndex
    planner: Planner


@dataclass
class Env:
    """Indexes and planners for every snapshot, shared by all systems (read-only)."""

    snapshots: dict[int, Snapshot]
    scope: str
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)

    @classmethod
    async def create(cls, corpora: dict[int, Corpus], scope: str) -> Env:
        snaps: dict[int, Snapshot] = {}
        embedder = HashingEmbedder()
        for k, corpus in corpora.items():
            ps = PassageSet(corpus)
            bm25 = BM25Index(ps)
            dense = await DenseIndex.build(ps, embedder)
            planner = Planner(PlannerConfig(bm25=bm25, dense=dense, access_scope=scope))
            snaps[k] = Snapshot(corpus, ps, bm25, dense, planner)
        return cls(snaps, scope, embedder)

    def operators(self) -> list[Operator[Any]]:
        ops: list[Operator[Any]] = [RRFOperator(), MergeOperator()]
        for s in self.snapshots.values():
            ops += [BM25Operator(s.bm25), DenseOperator(s.dense)]
        return ops


@dataclass
class WorldState:
    snapshot: int
    policy: AccessPolicy


@dataclass(frozen=True)
class TurnOutcome:
    evidence: tuple[Evidence, ...]
    executed_nodes: int
    reused_nodes: int
    retrieval_ops: int  # executed bm25/dense nodes
    seconds: float


class System(Protocol):
    name: str

    async def reset(self) -> None: ...

    async def turn(self, intent: Intent, state: WorldState) -> TurnOutcome: ...


def new_policy(scope: str) -> AccessPolicy:
    p = AccessPolicy()
    p.grant(PRINCIPAL, scope)
    return p


# -- deltaquery and full re-execution ------------------------------------------------------


class DeltaQuerySystem:
    def __init__(
        self,
        env: Env,
        *,
        reuse: bool = True,
        name: str = "deltaquery",
        precise_access: bool = True,
        cache_size: int = 4096,
    ) -> None:
        self.env = env
        self.reuse = reuse
        self.name = name
        self.precise_access = precise_access
        self.cache_size = cache_size
        self._store: EvidenceStore | None = None

    async def reset(self) -> None:
        if self._store is not None:
            await self._store.close()
        self._store = EvidenceStore(":memory:", cache_size=self.cache_size)

    async def turn(self, intent: Intent, state: WorldState) -> TurnOutcome:
        assert self._store is not None
        plan = self.env.snapshots[state.snapshot].planner.build(intent)
        ex = Executor(
            self._store, self.env.operators(), state.policy, precise_access=self.precise_access
        )
        t0 = time.perf_counter()
        r = await ex.execute(plan, principal=PRINCIPAL, reuse=self.reuse)
        dt = time.perf_counter() - t0
        executed = [t for t in r.trace if t.status is NodeStatus.EXECUTED]
        return TurnOutcome(
            evidence=r.output.evidence,
            executed_nodes=len(executed),
            reused_nodes=len(r.trace) - len(executed),
            retrieval_ops=sum(t.operator in RETRIEVERS for t in executed),
            seconds=dt,
        )


# -- generic in-memory operator caches (exact baselines and ablations) ---------------------


@dataclass(frozen=True)
class CacheKeyConfig:
    inputs: bool = True  # input artifact content hashes
    versions: bool = True  # snapshot / index / model versions
    # "none": access policy not in the key; "scope": the scope's whole policy version;
    # "precise": only the revocations the operator's own result can depend on.
    policy: str = "none"


def _registry(ops: Sequence[Operator[Any]]) -> dict[tuple[Any, ...], Operator[Any]]:
    out = {}
    for op in ops:
        b = op.bound_versions
        out[(op.name, b.source_snapshot, b.index_version, b.model_version)] = op
    return out


async def run_plan_cached(
    plan: Plan,
    registry: dict[tuple[Any, ...], Operator[Any]],
    cache: dict[str, tuple[Evidence, ...]] | None,
    cfg: CacheKeyConfig,
    policy: AccessPolicy,
) -> tuple[Artifact, int, int, int]:
    """Execute `plan`, reusing from `cache` by the configured key. Returns
    (output artifact, executed nodes, reused nodes, executed retrieval ops)."""
    now = datetime.now(UTC)
    done: dict[str, Artifact] = {}
    executed = reused = retrieval = 0

    async def resolve(node: Node) -> tuple[Artifact, bool]:
        op = registry[(node.operator, node.source_snapshot, node.index_version, node.model_version)]
        inputs = [done[i] for i in node.input_ids]
        key_parts: dict[str, Any] = {
            "op": node.operator,
            "params": node.canonical_parameters,
            "scope": node.access_scope,
        }
        if cfg.inputs:
            key_parts["inputs"] = [a.artifact_hash for a in inputs]
        if cfg.versions:
            key_parts["versions"] = [node.source_snapshot, node.index_version, node.model_version]
        if cfg.policy == "scope":
            key_parts["policy"] = policy.version(node.access_scope)
        elif cfg.policy == "precise":
            deps = op.access_dependencies(
                op.parse_params(node.canonical_parameters),
                node.access_scope,
                policy.revoked_sources(node.access_scope),
            )
            key_parts["policy"] = sorted(deps)
        key = hash_canonical("bench-cache", key_parts)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            ev = hit
            was_reused = True
        else:
            ctx = ExecutionContext(
                signature=key,
                access_scope=node.access_scope,
                now=now,
                excluded_sources=policy.revoked_sources(node.access_scope),
            )
            ev = tuple(await op.run(op.parse_params(node.canonical_parameters), inputs, ctx))
            if cache is not None:
                cache[key] = ev
            was_reused = False
        art = Artifact(
            signature=key,
            operator=op.name,
            operator_version=op.version,
            output_contract=op.output_contract,
            access_scope=node.access_scope,
            access_policy_version="n/a",
            source_snapshot=node.source_snapshot,
            index_version=node.index_version,
            model_version=node.model_version,
            input_signatures=tuple(a.signature for a in inputs),
            evidence=ev,
            artifact_hash=compute_artifact_hash(op.output_contract, ev),
            created_at=now,
        )
        return art, was_reused

    for layer in plan.layers():
        results = await asyncio.gather(*(resolve(n) for n in layer))
        for node, (art, was_reused) in zip(layer, results, strict=True):
            done[node.id] = art
            if was_reused:
                reused += 1
            else:
                executed += 1
                retrieval += node.operator in RETRIEVERS
    return done[plan.output_id], executed, reused, retrieval


class CacheSystem:
    def __init__(self, env: Env, name: str, cfg: CacheKeyConfig) -> None:
        self.env = env
        self.name = name
        self.cfg = cfg
        self.registry = _registry(env.operators())
        self.cache: dict[str, tuple[Evidence, ...]] = {}

    async def reset(self) -> None:
        self.cache = {}

    async def turn(self, intent: Intent, state: WorldState) -> TurnOutcome:
        plan = self.env.snapshots[state.snapshot].planner.build(intent)
        t0 = time.perf_counter()
        out, ex, ru, rt = await run_plan_cached(
            plan, self.registry, self.cache, self.cfg, state.policy
        )
        return TurnOutcome(out.evidence, ex, ru, rt, time.perf_counter() - t0)


# -- semantic (answer-level) cache ----------------------------------------------------------


class SemanticCacheSystem:
    def __init__(self, env: Env, threshold: float = 0.95, name: str | None = None) -> None:
        self.env = env
        self.threshold = threshold
        self.name = name or f"semantic@{threshold:g}"
        self.registry = _registry(env.operators())
        self.entries: list[tuple[tuple[float, ...], tuple[Evidence, ...]]] = []

    async def reset(self) -> None:
        self.entries = []

    async def turn(self, intent: Intent, state: WorldState) -> TurnOutcome:
        t0 = time.perf_counter()
        (q,) = await self.env.embedder.embed([render_question(intent)])
        q = l2_normalize(q)
        best, best_ev = -1.0, None
        for v, ev in self.entries:
            s = math.fsum(a * b for a, b in zip(q, v, strict=True))
            if s > best:
                best, best_ev = s, ev
        if best_ev is not None and best >= self.threshold:
            n = len(self.env.snapshots[state.snapshot].planner.build(intent).nodes)
            return TurnOutcome(best_ev, 0, n, 0, time.perf_counter() - t0)
        plan = self.env.snapshots[state.snapshot].planner.build(intent)
        out, ex, ru, rt = await run_plan_cached(
            plan, self.registry, None, CacheKeyConfig(), state.policy
        )
        self.entries.append((q, out.evidence))
        return TurnOutcome(out.evidence, ex, ru, rt, time.perf_counter() - t0)


# -- conversational hybrid RAG (no planner) -------------------------------------------------


class ConversationalHybridSystem:
    name = "conv_hybrid"

    def __init__(self, env: Env, per_slot_k: int = 3) -> None:
        self.env = env
        self.per_slot_k = per_slot_k
        self.registry = _registry(env.operators())

    async def reset(self) -> None:
        return None

    async def turn(self, intent: Intent, state: WorldState) -> TurnOutcome:
        s = self.env.snapshots[state.snapshot]
        q = render_question(intent)
        slots = len(intent.entities) * (len(intent.dimensions) + len(intent.constraints))
        k = min(100, self.per_slot_k * slots)
        common: dict[str, Any] = {
            "access_scope": self.env.scope,
            "output_contract": "ranked_passages",
        }
        plan = Plan(
            nodes=(
                Node(
                    id="lex",
                    operator="bm25",
                    canonical_parameters=BM25Params(query=q, top_k=k).model_dump(mode="json"),
                    source_snapshot=s.bm25.snapshot_id,
                    index_version=s.bm25.version,
                    model_version=None,
                    **common,
                ),
                Node(
                    id="dense",
                    operator="dense",
                    canonical_parameters=DenseParams(query=q, top_k=k).model_dump(mode="json"),
                    source_snapshot=s.dense.snapshot_id,
                    index_version=s.dense.version,
                    model_version=s.dense.model_version,
                    **common,
                ),
                Node(
                    id="fused",
                    operator="rrf",
                    canonical_parameters=RRFParams(top_k=k).model_dump(mode="json"),
                    input_ids=("lex", "dense"),
                    source_snapshot=None,
                    index_version=None,
                    model_version=None,
                    **common,
                ),
            ),
            output_id="fused",
        )
        t0 = time.perf_counter()
        out, ex, ru, rt = await run_plan_cached(
            plan, self.registry, None, CacheKeyConfig(), state.policy
        )
        return TurnOutcome(out.evidence, ex, ru, rt, time.perf_counter() - t0)


def all_systems(env: Env) -> list[System]:
    return [
        DeltaQuerySystem(env),
        DeltaQuerySystem(env, reuse=False, name="full"),
        DeltaQuerySystem(env, name="dq_coarse_policy", precise_access=False),
        DeltaQuerySystem(env, name="dq_no_memcache", cache_size=0),
        CacheSystem(env, "exact_naive", CacheKeyConfig(inputs=True, versions=False)),
        CacheSystem(env, "exact_versioned", CacheKeyConfig(inputs=True, versions=True)),
        SemanticCacheSystem(env, 0.95),
        ConversationalHybridSystem(env),
        CacheSystem(
            env, "abl_no_deps", CacheKeyConfig(inputs=False, versions=True, policy="precise")
        ),
        CacheSystem(
            env, "abl_no_versions", CacheKeyConfig(inputs=True, versions=False, policy="precise")
        ),
        CacheSystem(
            env, "abl_dq_keys", CacheKeyConfig(inputs=True, versions=True, policy="precise")
        ),
    ]
