"""DeltaBench runners.

Oracle track: every system gets the gold intent of every step, so differences are purely about
reuse. Each sequence starts cold (fresh cache/store, snapshot 0, fresh policy) unless `warm`.
Rows are appended to a JSONL log (never rewritten); the report is derived from the log.

E2E track: DeltaQuery only, with the real LLM compiler and answer generator. Measures compiler
exact-match against gold intents, answer fact recall and citation precision.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from collections.abc import Sequence as Seq
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deltaquery.compiler.compiler import CompileError, IntentCompiler
from deltaquery.compiler.intent import Attribute, Dimension, Intent
from deltaquery.eval.synth import (
    DATASET_VERSION,
    SCOPE,
    Sequence,
    World,
    gold_keys,
)
from deltaquery.eval.systems import (
    PRINCIPAL,
    Env,
    System,
    WorldState,
    all_systems,
    new_policy,
)
from deltaquery.evidence.models import Evidence
from deltaquery.evidence.store import EvidenceStore
from deltaquery.llm.base import StructuredLLM
from deltaquery.runtime.answer import AnswerGenerator, SpanResolver
from deltaquery.runtime.executor import Executor
from deltaquery.session import Session, SupersededError


def _versions(env: Env) -> dict[int, dict[str, str]]:
    return {
        k: {d.source_id: d.source_version for d in s.corpus.documents}
        for k, s in env.snapshots.items()
    }


def score_evidence(
    evidence: Seq[Evidence],
    gold: set[tuple[str, int, int]],
    current_versions: dict[str, str],
    revoked: frozenset[str],
) -> dict[str, Any]:
    valid: set[tuple[str, int, int]] = set()
    stale = 0
    for e in evidence:
        key = (e.source_id, e.span.start, e.span.end)
        if e.source_id in revoked or current_versions.get(e.source_id) != e.source_version:
            stale += 1
        else:
            valid.add(key)
    return {
        "recall": (len(gold & valid) / len(gold)) if gold else None,
        "stale": stale > 0,
        "stale_count": stale,
        "n_evidence": len(evidence),
        "n_gold": len(gold),
    }


async def run_oracle(
    world: World,
    sequences: Seq[Sequence],
    out_path: Path,
    *,
    systems: Seq[str] | None = None,
    trials: int = 3,
    seed: int = 0,
    warm: bool = False,
    planner_variant: str = "A",
) -> str:
    env = await Env.create({0: world.corpus(0), 1: world.corpus(1)}, SCOPE, planner_variant)
    versions = _versions(env)
    chosen: list[System] = [s for s in all_systems(env) if systems is None or s.name in systems]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    gold_cache: dict[tuple[int, str, frozenset[str]], set[tuple[str, int, int]]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "track": "oracle",
        "dataset": DATASET_VERSION,
        "world_seed": world.seed,
        "n_sequences": len(sequences),
        "trials": trials,
        "warm": warm,
        "planner_variant": planner_variant,
        "splits": sorted({s.split for s in sequences}),
        "systems": [s.name for s in chosen],
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for trial in range(trials):
            order = list(chosen)
            random.Random(seed * 1000 + trial).shuffle(order)  # vary system order per trial
            for system in order:
                if warm:
                    await system.reset()
                for seq in sequences:
                    if not warm:
                        await system.reset()
                    state = WorldState(snapshot=0, policy=new_policy(SCOPE))
                    for i, step in enumerate(seq.steps):
                        if step.event and "snapshot" in step.event:
                            state.snapshot = int(step.event["snapshot"])
                        if step.event and "revoke" in step.event:
                            state.policy.revoke_source(SCOPE, step.event["revoke"])
                        intent = Intent.model_validate(step.intent)
                        out = await system.turn(intent, state)
                        revoked = state.policy.revoked_sources(SCOPE)
                        gk = (state.snapshot, json.dumps(step.intent, sort_keys=True), revoked)
                        if gk not in gold_cache:
                            gold_cache[gk] = gold_keys(
                                world, env.snapshots[state.snapshot].corpus, intent, revoked
                            )
                        row = {
                            "run_id": run_id,
                            "trial": trial,
                            "system": system.name,
                            "seq": seq.id,
                            "split": seq.split,
                            "step": i,
                            "kind": step.kind,
                            "snapshot": state.snapshot,
                            "executed_nodes": out.executed_nodes,
                            "reused_nodes": out.reused_nodes,
                            "retrieval_ops": out.retrieval_ops,
                            "seconds": out.seconds,
                            **score_evidence(
                                out.evidence, gold_cache[gk], versions[state.snapshot], revoked
                            ),
                        }
                        f.write(json.dumps(row) + "\n")
    return run_id


# -- e2e track ------------------------------------------------------------------------------


def fact_patterns(world: World, snapshot: int, intent: Intent) -> list[re.Pattern[str]]:
    pats: list[re.Pattern[str]] = []
    for e in intent.entities:
        f = world.facts[snapshot][e]
        dims = set(intent.dimensions)
        attrs = {c.attribute for c in intent.constraints}
        if Dimension.DISASTER_RECOVERY in dims or Attribute.RPO in attrs:
            alt = "|zero" if f.rpo == 0 else ""
            pats.append(re.compile(rf"\b(?:{f.rpo}\s*(?:seconds?|s\b){alt})", re.I))
        if Dimension.DISASTER_RECOVERY in dims or Attribute.RTO in attrs:
            pats.append(re.compile(rf"\b{f.rto}\s*(?:seconds?|s\b)", re.I))
        if Dimension.PRICING in dims:
            pats.append(re.compile(re.escape(f.price)))
        if Dimension.BACKUP in dims:
            pats.append(re.compile(rf"\b{f.retention_days}\s*days?\b", re.I))
        if Dimension.REGIONS in dims:
            pats.append(re.compile(rf"\b{f.regions}\s*regions?\b", re.I))
        if Dimension.ENCRYPTION in dims:
            pats.append(re.compile(re.escape(f.key_mode), re.I))
    return pats


async def run_e2e(
    world: World,
    sequences: Seq[Sequence],
    llm: StructuredLLM,
    out_path: Path,
) -> str:
    env = await Env.create({0: world.corpus(0), 1: world.corpus(1)}, SCOPE)
    run_id = "e2e-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entities = world.entity_ids
    with out_path.open("a", encoding="utf-8") as f:
        meta = {
            "run_id": run_id,
            "track": "e2e",
            "dataset": DATASET_VERSION,
            "model": llm.model_version,
            "n_sequences": len(sequences),
        }
        f.write(json.dumps({"meta": meta}) + "\n")
        for seq in sequences:
            policy = new_policy(SCOPE)
            snapshot = 0
            store = EvidenceStore(":memory:")
            executor = Executor(store, env.operators(), policy)
            gen = AnswerGenerator(llm, SpanResolver(env.snapshots[0].passages))
            session = Session(
                IntentCompiler(llm, entities),
                env.snapshots[0].planner,
                executor,
                principal=PRINCIPAL,
                generator=gen,
            )
            for i, step in enumerate(seq.steps):
                if step.event and "snapshot" in step.event:
                    snapshot = int(step.event["snapshot"])
                    session.planner = env.snapshots[snapshot].planner
                    gen.resolver = SpanResolver(env.snapshots[snapshot].passages)
                if step.event and "revoke" in step.event:
                    policy.revoke_source(SCOPE, step.event["revoke"])
                gold_intent = Intent.model_validate(step.intent)
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "seq": seq.id,
                    "split": seq.split,
                    "step": i,
                    "kind": step.kind,
                    "text": step.text,
                    "gold_intent": step.intent,
                }
                t0 = time.perf_counter()
                try:
                    turn = await session.ask(step.text)
                except (CompileError, SupersededError) as exc:
                    row |= {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    f.write(json.dumps(row) + "\n")
                    continue
                revoked = policy.revoked_sources(SCOPE)
                gold = gold_keys(world, env.snapshots[snapshot].corpus, gold_intent, revoked)
                ans = turn.answer
                cites = [c for s in (ans.sentences if ans else ()) for c in s.citations]
                pats = fact_patterns(world, snapshot, gold_intent)
                text = ans.text if ans and not ans.abstained else ""
                row |= {
                    "pred_intent": turn.intent.model_dump(mode="json"),
                    "intent_exact": turn.intent == gold_intent,
                    "pred_ops": [o.kind.value for o in turn.compiled.ops],
                    "abstained": bool(ans and ans.abstained),
                    "fact_recall": (sum(bool(p.search(text)) for p in pats) / len(pats))
                    if pats
                    else None,
                    "citation_precision": (
                        sum((c.source_id, c.span.start, c.span.end) in gold for c in cites)
                        / len(cites)
                    )
                    if cites
                    else None,
                    "executed_nodes": len(turn.trace.executed_nodes),
                    "reused_nodes": len(turn.trace.reused_nodes),
                    "timing_s": turn.trace.to_dict()["timing_s"],
                    "wall_s": time.perf_counter() - t0,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
            await store.close()
    return run_id


# -- compile-only track ---------------------------------------------------------------------


async def run_compile(
    world: World,
    sequences: Seq[Sequence],
    llm: StructuredLLM,
    out_path: Path,
    *,
    grounding: bool = True,
) -> str:
    """Intent accuracy of the real compiler, no retrieval or generation. Each turn is compiled
    against the *committed* prior (the compiler's own previous output), as in a session, so
    errors propagate the same way they do end to end."""
    tag = "grounded" if grounding else "ungrounded"
    run_id = f"compile-{tag}-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id += "-" + uuid.uuid4().hex[:6]
    compiler = IntentCompiler(llm, world.entity_ids, grounding=grounding)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        meta = {
            "run_id": run_id,
            "track": "compile",
            "dataset": DATASET_VERSION,
            "model": llm.model_version,
            "grounding": grounding,
            "n_sequences": len(sequences),
        }
        f.write(json.dumps({"meta": meta}) + "\n")
        for seq in sequences:
            prior: Intent | None = None
            for i, step in enumerate(seq.steps):
                gold = Intent.model_validate(step.intent)
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "seq": seq.id,
                    "split": seq.split,
                    "step": i,
                    "kind": step.kind,
                    "text": step.text,
                    "gold_intent": step.intent,
                }
                t0 = time.perf_counter()
                try:
                    r = await compiler.compile(step.text, prior)
                except CompileError as exc:
                    row |= {"error": str(exc)[:300], "seconds": time.perf_counter() - t0}
                    f.write(json.dumps(row) + "\n")
                    continue
                prior = r.intent
                row |= {
                    "pred_intent": r.intent.model_dump(mode="json"),
                    "intent_exact": r.intent == gold,
                    "llm_called": r.attempts > 0,
                    "notes": list(r.grounding_notes),
                    "seconds": time.perf_counter() - t0,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
    return run_id
