"""DeltaBench CLI.

python -m deltaquery.eval generate [--n 300]           write dataset files to bench/
python -m deltaquery.eval oracle [--trials 3] [--n N]   run oracle track, append log, report
python -m deltaquery.eval e2e --n 10 [--model M]        run e2e track (local Ollama), report
python -m deltaquery.eval report LOG [--run-id ID]      regenerate a report from a log
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from deltaquery.eval.report import write_report
from deltaquery.eval.runner import run_e2e, run_oracle
from deltaquery.eval.synth import DATASET_VERSION, build_world, generate_sequences

BENCH = Path("bench") / DATASET_VERSION
RESULTS = Path("results") / "deltabench"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(n: int, world_seed: int, seq_seed: int) -> None:
    world = build_world(world_seed)
    seqs = generate_sequences(world, n, seq_seed)
    BENCH.mkdir(parents=True, exist_ok=True)
    for snap in (0, 1):
        docs = [d.model_dump(mode="json") for d in world.corpus(snap).documents]
        (BENCH / f"corpus_v{snap}.json").write_text(json.dumps(docs, indent=1), encoding="utf-8")
    with (BENCH / "sequences.jsonl").open("w", encoding="utf-8") as f:
        for s in seqs:
            f.write(s.model_dump_json() + "\n")
    manifest = {
        "dataset": DATASET_VERSION,
        "world_seed": world_seed,
        "sequence_seed": seq_seed,
        "n_sequences": n,
        "changed_entities_v1": sorted(world.changed_entities),
        "sha256": {p.name: _sha(p) for p in sorted(BENCH.glob("*")) if p.name != "manifest.json"},
    }
    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {BENCH} ({n} sequences, {sum(len(s.steps) for s in seqs)} turns)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m deltaquery.eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=300)
    for p in (g,):
        p.add_argument("--world-seed", type=int, default=7)
        p.add_argument("--seq-seed", type=int, default=11)
    o = sub.add_parser("oracle")
    o.add_argument("--n", type=int, default=300)
    o.add_argument("--trials", type=int, default=3)
    o.add_argument("--warm", action="store_true")
    o.add_argument("--systems", nargs="*")
    o.add_argument("--world-seed", type=int, default=7)
    o.add_argument("--seq-seed", type=int, default=11)
    e = sub.add_parser("e2e")
    e.add_argument("--n", type=int, default=10)
    e.add_argument("--model", default="qwen3.5:4b")
    e.add_argument("--world-seed", type=int, default=7)
    e.add_argument("--seq-seed", type=int, default=11)
    r = sub.add_parser("report")
    r.add_argument("log", type=Path)
    r.add_argument("--run-id")
    args = ap.parse_args()

    if args.cmd == "generate":
        generate(args.n, args.world_seed, args.seq_seed)
        return
    if args.cmd == "report":
        out = args.log.with_name(f"report-{args.run_id or 'latest'}.md")
        print(write_report(args.log, out, args.run_id))
        return
    world = build_world(args.world_seed)
    seqs = generate_sequences(world, max(args.n, 1), args.seq_seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "oracle":
        log = RESULTS / ("oracle-warm.jsonl" if args.warm else "oracle.jsonl")
        run_id = asyncio.run(
            run_oracle(world, seqs, log, systems=args.systems, trials=args.trials, warm=args.warm)
        )
    else:
        from deltaquery.llm.ollama import OllamaLLM

        log = RESULTS / "e2e.jsonl"

        async def go() -> str:
            llm = await OllamaLLM.connect(args.model)
            return await run_e2e(world, seqs, llm, log)

        run_id = asyncio.run(go())
    out = write_report(log, RESULTS / f"report-{run_id}.md", run_id)
    print(f"run {run_id}\nlog {log}\nreport {out}")


if __name__ == "__main__":
    main()
