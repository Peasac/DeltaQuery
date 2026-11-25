"""Markdown report from a DeltaBench JSONL log. Pure stdlib; regenerable at any time.

Uncertainty: 95% percentile-bootstrap intervals over *sequences* (the unit that is sampled),
with per-sequence values averaged over trials first.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

DECISIVE = "exact_versioned"


def load(path: Path, run_id: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metas: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "meta" in rec:
            metas[rec["meta"]["run_id"]] = rec["meta"]
        else:
            rows.append(rec)
    if run_id is None:
        run_id = list(metas)[-1]
    return metas[run_id], [r for r in rows if r["run_id"] == run_id]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def bootstrap_ratio(
    per_seq_a: dict[str, float], per_seq_b: dict[str, float], b: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Ratio of sums a/b with a 95% bootstrap CI over sequences."""
    keys = sorted(set(per_seq_a) & set(per_seq_b))
    rng = random.Random(seed)

    def ratio(ks: Iterable[str]) -> float:
        ks = list(ks)
        den = sum(per_seq_b[k] for k in ks)
        return sum(per_seq_a[k] for k in ks) / den if den else float("nan")

    point = ratio(keys)
    samples = [ratio(rng.choices(keys, k=len(keys))) for _ in range(b)]
    return point, _pct(samples, 0.025), _pct(samples, 0.975)


def bootstrap_mean(
    values: dict[str, float], b: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    keys = sorted(values)
    rng = random.Random(seed)
    point = statistics.fmean(values[k] for k in keys)
    samples = [
        statistics.fmean(values[k] for k in rng.choices(keys, k=len(keys))) for _ in range(b)
    ]
    return point, _pct(samples, 0.025), _pct(samples, 0.975)


def per_seq(
    rows: list[dict[str, Any]], system: str, fn: Callable[[dict[str, Any]], float]
) -> dict[str, float]:
    """Per-sequence total of fn(row), averaged over trials."""
    acc: dict[tuple[str, int], float] = defaultdict(float)
    for r in rows:
        if r["system"] == system:
            acc[(r["seq"], r["trial"])] += fn(r)
    by_seq: dict[str, list[float]] = defaultdict(list)
    for (seq, _), v in acc.items():
        by_seq[seq].append(v)
    return {k: statistics.fmean(v) for k, v in by_seq.items()}


def render(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    systems = list(dict.fromkeys(r["system"] for r in rows))
    by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    n_turns = len(by_sys[systems[0]]) // max(1, meta["trials"])
    out: list[str] = []
    w = out.append
    w(f"# DeltaBench report: {meta['run_id']}\n")
    w(
        f"Dataset `{meta['dataset']}` (synthetic, world seed {meta['world_seed']}), "
        f"{meta['n_sequences']} sequences, {n_turns} turns per system per trial, "
        f"{meta['trials']} trials, {'warm' if meta['warm'] else 'cold'} start per sequence. "
        "Oracle-intent track: every system receives the gold intent, so differences are about "
        "reuse only. Latency is retrieval-side wall time on one machine; it includes each "
        "system's own bookkeeping (SQLite store for `deltaquery`/`full`, in-memory dicts for the "
        "cache baselines).\n"
    )
    w("## Overall\n")
    w(
        "| system | retrieval ops / turn | executed nodes / turn | p50 ms | p95 ms | "
        "evidence recall | stale-turn rate |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    for s in systems:
        rs = by_sys[s]
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        ms = [r["seconds"] * 1000 for r in rs]
        w(
            f"| {s} | {statistics.fmean(r['retrieval_ops'] for r in rs):.2f} | "
            f"{statistics.fmean(r['executed_nodes'] for r in rs):.2f} | "
            f"{_pct(ms, 0.5):.1f} | {_pct(ms, 0.95):.1f} | "
            f"{statistics.fmean(rec):.3f} | {statistics.fmean(r['stale'] for r in rs):.3f} |"
        )

    def ops(r: dict[str, Any]) -> float:
        return float(r["retrieval_ops"])

    def secs(r: dict[str, Any]) -> float:
        return float(r["seconds"])

    w("\n## Ratios with 95% bootstrap CIs (over sequences)\n")
    w("| comparison | retrieval ops ratio | latency ratio |")
    w("|---|---|---|")
    pairs = [("deltaquery", "full"), ("deltaquery", DECISIVE), ("deltaquery", "exact_naive"),
             ("abl_dq_keys", DECISIVE), ("deltaquery", "abl_dq_keys"),
             ("deltaquery", "dq_coarse_policy"), ("deltaquery", "dq_no_memcache")]  # fmt: skip
    for a, bsys in pairs:
        if a in by_sys and bsys in by_sys:
            ro = bootstrap_ratio(per_seq(rows, a, ops), per_seq(rows, bsys, ops))
            rl = bootstrap_ratio(per_seq(rows, a, secs), per_seq(rows, bsys, secs))
            w(
                f"| {a} / {bsys} | {ro[0]:.3f} [{ro[1]:.3f}, {ro[2]:.3f}] | "
                f"{rl[0]:.3f} [{rl[1]:.3f}, {rl[2]:.3f}] |"
            )

    w("\n## Stale-turn rate and recall by revision kind\n")
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    w("Stale-turn rate (fraction of turns whose output cites an outdated or revoked source):\n")
    w("| kind | n | " + " | ".join(systems) + " |")
    w("|---|---:|" + "---:|" * len(systems))
    for k in kinds:
        cells = []
        n = 0
        for s in systems:
            rk = [r for r in by_sys[s] if r["kind"] == k]
            n = len(rk) // max(1, meta["trials"])
            cells.append(f"{statistics.fmean(r['stale'] for r in rk):.3f}")
        w(f"| {k} | {n} | " + " | ".join(cells) + " |")
    w("\nRetrieval ops per turn:\n")
    w("| kind | " + " | ".join(systems) + " |")
    w("|---|" + "---:|" * len(systems))
    for k in kinds:
        cells = [
            f"{statistics.fmean(r['retrieval_ops'] for r in by_sys[s] if r['kind'] == k):.2f}"
            for s in systems
        ]
        w(f"| {k} | " + " | ".join(cells) + " |")
    w("\nEvidence recall:\n")
    w("| kind | " + " | ".join(systems) + " |")
    w("|---|" + "---:|" * len(systems))
    for k in kinds:
        cells = []
        for s in systems:
            rec = [r["recall"] for r in by_sys[s] if r["kind"] == k and r["recall"] is not None]
            cells.append(f"{statistics.fmean(rec):.3f}" if rec else "n/a")
        w(f"| {k} | " + " | ".join(cells) + " |")

    w("\n## Split\n")
    w("| system | dev recall | test recall | dev stale | test stale |")
    w("|---|---:|---:|---:|---:|")
    for s in systems:
        cells = []
        for metric in ("recall", "stale"):
            for split in ("dev", "test"):
                v = [
                    float(r[metric])
                    for r in by_sys[s]
                    if r["split"] == split and r[metric] is not None
                ]
                cells.append(f"{statistics.fmean(v):.3f}" if v else "n/a")
        w(f"| {s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    if "deltaquery" in by_sys and DECISIVE in by_sys:
        w("\n## Go / no-go reading (project.md §18)\n")
        ro = bootstrap_ratio(per_seq(rows, "deltaquery", ops), per_seq(rows, DECISIVE, ops))
        st_dq = bootstrap_mean(per_seq(rows, "deltaquery", lambda r: float(r["stale"])))
        st_ev = bootstrap_mean(per_seq(rows, DECISIVE, lambda r: float(r["stale"])))
        cost = "fewer" if ro[2] < 1 else "more" if ro[1] > 1 else "statistically indistinguishable"
        w(
            f"- Cost: DeltaQuery uses {cost} retrieval ops than `{DECISIVE}` "
            f"(ratio {ro[0]:.3f}, 95% CI [{ro[1]:.3f}, {ro[2]:.3f}])."
        )
        w(
            f"- Correctness: stale turns per sequence, DeltaQuery {st_dq[0]:.3f} "
            f"[{st_dq[1]:.3f}, {st_dq[2]:.3f}] vs `{DECISIVE}` {st_ev[0]:.3f} "
            f"[{st_ev[1]:.3f}, {st_ev[2]:.3f}]."
        )
        w(
            "- Per §18, a paper claim needs a *cost* win over exact operator caching at preserved "
            "quality. Any advantage shown here that comes only from stale-reuse avoidance is a "
            "correctness/safety result, not a cost result. Synthetic data; external validity "
            "requires a real dataset."
        )
    return "\n".join(out) + "\n"


def render_e2e(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    w = out.append
    ok = [r for r in rows if "error" not in r]
    w(f"# DeltaBench e2e report: {meta['run_id']}\n")
    w(
        f"Model `{meta['model']}`, {meta['n_sequences']} sequences, {len(rows)} turns "
        f"({len(rows) - len(ok)} failed to compile or were superseded). Single run; "
        "temperature 0. Compiler errors propagate (later turns revise the committed intent).\n"
    )

    def mean(xs: list[Any]) -> str:
        xs = [x for x in xs if x is not None]
        return f"{statistics.fmean(xs):.3f}" if xs else "n/a"

    w(
        "Intent exact match compares the whole intent. Plan-equivalent match ignores `task`, "
        "which the planner does not use, so it measures whether the *plan* was right.\n"
    )
    w(
        "| kind | n | intent exact match | plan-equivalent match | fact recall | "
        "citation precision | abstain rate |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    for k in [*kinds, "ALL"]:
        rk = [r for r in ok if k == "ALL" or r["kind"] == k]
        n_all = len([r for r in rows if k == "ALL" or r["kind"] == k])
        w(
            f"| {k} | {n_all} | {mean([float(r['intent_exact']) for r in rk])} | "
            f"{mean([float(_plan_equal(r)) for r in rk])} | "
            f"{mean([r['fact_recall'] for r in rk])} | "
            f"{mean([r['citation_precision'] for r in rk])} | "
            f"{mean([float(r['abstained']) for r in rk])} |"
        )
    if ok:
        comp = [r["timing_s"]["compile"] for r in ok]
        retr = [r["timing_s"]["retrieval"] for r in ok]
        gen = [r["timing_s"]["generation"] for r in ok]
        w(
            f"\nMedian seconds per turn: compile {statistics.median(comp):.2f}, retrieval "
            f"{statistics.median(retr):.3f}, generation {statistics.median(gen):.2f}."
        )
    return "\n".join(out) + "\n"


def _plan_equal(r: dict[str, Any]) -> bool:
    g, p = r["gold_intent"], r["pred_intent"]
    return all(g[k] == p[k] for k in ("entities", "dimensions", "constraints"))


def write_report(log: Path, out: Path, run_id: str | None = None) -> Path:
    meta, rows = load(log, run_id)
    text = render_e2e(meta, rows) if meta.get("track") == "e2e" else render(meta, rows)
    out.write_text(text, encoding="utf-8")
    return out
