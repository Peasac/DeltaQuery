"""DeltaBench harness: generator determinism, gold labels, metric definitions, system sanity."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from deltaquery.compiler.intent import Intent
from deltaquery.eval.report import load, render
from deltaquery.eval.runner import fact_patterns, run_oracle, score_evidence
from deltaquery.eval.synth import (
    GOLD_PARAGRAPHS,
    SCOPE,
    build_world,
    generate_sequences,
    gold_keys,
    source_id,
)


@pytest.fixture(scope="module")
def world() -> Any:
    return build_world(7)


def test_world_is_deterministic_and_v1_changes_only_changed_entities(world: Any) -> None:
    again = build_world(7)
    assert world.corpus(0).snapshot_id == again.corpus(0).snapshot_id
    v0 = {d.source_id: d.source_version for d in world.corpus(0).documents}
    v1 = {d.source_id: d.source_version for d in world.corpus(1).documents}
    changed = {s for s in v0 if v0[s] != v1[s]}
    expected = {
        f"{e}-{d}" for e in world.changed_entities for d in ("disaster-recovery", "pricing")
    }
    assert changed == expected and changed


def test_sequences_are_deterministic_and_valid(world: Any) -> None:
    a = generate_sequences(world, 30, 11)
    b = generate_sequences(world, 30, 11)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    for s in a:
        assert s.steps[0].kind == "new" and 4 <= len(s.steps) <= 6
        pool = set(world.entity_ids[:20] if s.split == "dev" else world.entity_ids[20:])
        for st in s.steps:
            it = Intent.model_validate(st.intent)
            assert set(it.entities) <= pool  # held-out split by entity family
        assert sum(st.kind == "source_update" for st in s.steps) <= 1


def test_gold_passages_contain_the_facts(world: Any) -> None:
    from deltaquery.compiler.intent import Dimension

    corpus = world.corpus(0)
    docs = {d.source_id: d for d in corpus.documents}
    e = world.entity_ids[0]
    it = Intent.model_validate(
        {"task": "describe", "entities": [e], "dimensions": ["disaster_recovery", "pricing"]}
    )
    gold = gold_keys(world, corpus, it, frozenset())
    texts = {docs[s].text[a:b] for s, a, b in gold}
    f = world.facts[0][e]
    assert any(f"RPO is {f.rpo} seconds" in t for t in texts)
    assert any(f"{f.rto} seconds" in t for t in texts)
    assert any(f"${f.price}" in t for t in texts)
    n = sum(len(GOLD_PARAGRAPHS[d]) for d in (Dimension.DISASTER_RECOVERY, Dimension.PRICING))
    assert len(gold) == n
    revoked = frozenset({source_id(world.name_of(e), Dimension.PRICING)})
    assert len(gold_keys(world, corpus, it, revoked)) == n - 1


def test_score_evidence_flags_outdated_and_revoked() -> None:
    from datetime import UTC, datetime

    from deltaquery.evidence.models import CoverageType, Evidence, Span

    def ev(sid: str, ver: str) -> Evidence:
        return Evidence(
            source_id=sid, source_version=ver, span=Span(start=0, end=5), content_hash="h",
            retrieval_node_id="n", score=1.0, coverage_type=CoverageType.LEXICAL,
            created_at=datetime.now(UTC),
        )  # fmt: skip

    gold = {("a", 0, 5), ("b", 0, 5)}
    cur = {"a": "v2", "b": "v1", "c": "v1"}
    s = score_evidence([ev("a", "v1"), ev("b", "v1")], gold, cur, frozenset())
    assert s["stale"] and s["stale_count"] == 1 and s["recall"] == 0.5  # outdated is not recall
    s = score_evidence([ev("c", "v1")], gold, cur, frozenset({"c"}))
    assert s["stale"] and s["recall"] == 0.0


def test_fact_patterns(world: Any) -> None:
    e = world.entity_ids[0]
    it = Intent.model_validate({"task": "describe", "entities": [e], "dimensions": ["pricing"]})
    (p,) = fact_patterns(world, 0, it)
    assert p.search(f"costs ${world.facts[0][e].price} per GB")


def test_small_oracle_run_sanity(world: Any, tmp_path: Path) -> None:
    seqs = generate_sequences(world, 40, 11)
    log = tmp_path / "log.jsonl"
    asyncio.run(run_oracle(world, seqs, log, trials=1))
    meta, rows = load(log)
    by: dict[tuple[str, str, int], dict[str, Any]] = {}
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        by[(r["system"], r["seq"], r["step"])] = r
        totals[r["system"]]["stale"] += r["stale"]
        totals[r["system"]]["ops"] += r["retrieval_ops"]
    # Safety: deltaquery and full re-execution never cite stale or revoked evidence.
    assert totals["deltaquery"]["stale"] == 0 and totals["full"]["stale"] == 0
    # The naive and policy-unaware caches do (the scenarios exist in the data).
    assert totals["exact_naive"]["stale"] > 0 and totals["exact_versioned"]["stale"] > 0
    # Reuse never changes what deltaquery returns relative to full re-execution.
    for (sysname, seq, step), r in by.items():
        if sysname == "deltaquery":
            f = by[("full", seq, step)]
            assert r["recall"] == f["recall"] and r["n_evidence"] == f["n_evidence"]
            assert r["retrieval_ops"] <= f["retrieval_ops"]
    assert totals["deltaquery"]["ops"] < totals["full"]["ops"]
    text = render(meta, rows)
    assert "Go / no-go" in text and "exact_versioned" in text
    json.dumps(meta)
    assert SCOPE == "bench"
