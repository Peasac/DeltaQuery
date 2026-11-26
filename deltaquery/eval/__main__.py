"""DeltaBench CLI.

python -m deltaquery.eval generate [--n 300]           write dataset files to bench/
python -m deltaquery.eval oracle [--trials 3] [--n N]   run oracle track, append log, report
python -m deltaquery.eval e2e --n 10 [--model M]        run e2e track (local Ollama), report
python -m deltaquery.eval report LOG [--run-id ID]      regenerate a report from a log
python -m deltaquery.eval paraphrase-sheet --seq-seed 29 [--n 40] [--k 15] [--out PATH]
python -m deltaquery.eval compile --texts SHEET ...     compile track on reworded messages
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
    o.add_argument("--planner", default="A", choices=["A", "B", "C", "D"])
    o.add_argument("--split", default="all", choices=["all", "dev", "test"])
    o.add_argument("--systems", nargs="*")
    o.add_argument("--world-seed", type=int, default=7)
    o.add_argument("--seq-seed", type=int, default=11)
    e = sub.add_parser("e2e")
    e.add_argument("--n", type=int, default=10)
    e.add_argument("--model", default="qwen3.5:4b")
    e.add_argument("--world-seed", type=int, default=7)
    e.add_argument("--seq-seed", type=int, default=11)
    c = sub.add_parser("compile")
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--model", default="qwen3.5:4b")
    c.add_argument("--world-seed", type=int, default=7)
    c.add_argument("--seq-seed", type=int, default=11)
    c.add_argument(
        "--compare", action="store_true", help="run ungrounded and grounded on the same sequences"
    )
    c.add_argument("--texts", type=Path, help="filled paraphrase sheet replacing message texts")
    ps = sub.add_parser("paraphrase-sheet")
    ps.add_argument("--n", type=int, default=40)
    ps.add_argument("--k", type=int, default=15)
    ps.add_argument("--world-seed", type=int, default=7)
    ps.add_argument("--seq-seed", type=int, default=29)
    ps.add_argument("--out", type=Path, default=Path("bench") / "paraphrase-v1" / "sheet.md")
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
    if args.cmd == "paraphrase-sheet":
        from deltaquery.eval.paraphrase import build_sheet, select

        chosen = select(seqs, args.k)
        src = f"world seed {args.world_seed}, sequence seed {args.seq_seed}, first {args.n}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(build_sheet(world, chosen, src), encoding="utf-8")
        turns = sum(len(q.steps) for q in chosen)
        print(f"wrote {args.out} ({len(chosen)} conversations, {turns} messages)")
        return
    if args.cmd == "compile":
        from deltaquery.eval.report import write_compile_report
        from deltaquery.eval.runner import run_compile
        from deltaquery.llm.ollama import OllamaLLM

        log = RESULTS / "compile.jsonl"
        if args.texts is not None:
            from deltaquery.eval.paraphrase import apply_texts, parse_sheet

            seqs = apply_texts(seqs, parse_sheet(args.texts.read_text(encoding="utf-8")))
            if not seqs:
                raise SystemExit(f"{args.texts}: no fully answered conversation")
            log = RESULTS / "compile-paraphrase.jsonl"
            print(f"{len(seqs)} answered conversations from {args.texts}")

        async def go_compile() -> list[str]:
            llm = await OllamaLLM.connect(args.model)
            ids = []
            for g in [False, True] if args.compare else [True]:
                ids.append(await run_compile(world, seqs, llm, log, grounding=g))
            return ids

        ids = asyncio.run(go_compile())
        out = write_compile_report(log, RESULTS / f"report-{ids[-1]}.md", ids)
        print(f"runs {ids}\nlog {log}\nreport {out}")
        return
    if args.cmd == "oracle":
        log = RESULTS / ("oracle-warm.jsonl" if args.warm else "oracle.jsonl")
        if args.split != "all":
            seqs = [s for s in seqs if s.split == args.split]
        run_id = asyncio.run(
            run_oracle(
                world,
                seqs,
                log,
                systems=args.systems,
                trials=args.trials,
                warm=args.warm,
                planner_variant=args.planner,
            )
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
