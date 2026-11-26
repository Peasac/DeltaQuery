"""Paraphrase track: the same sequences and gold intents, with messages reworded by a person.

`build_sheet` writes a fill-in sheet that describes what each message must *mean* (from the
gold intents) without showing the benchmark's own wording, so the writer is not anchored on
the templates the grounding keyword lists were built from. `parse_sheet` reads the answers
back; `apply_texts` swaps them into the sequences. Only fully answered sequences are used.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence as Seq

from deltaquery.compiler.diff import RevisionKind, diff_intents
from deltaquery.compiler.intent import Intent
from deltaquery.eval.synth import Sequence, Step, World

SHEET_VERSION = "paraphrase-sheet-v1"
_TOPIC = {
    "disaster_recovery": "disaster recovery",
    "pricing": "pricing",
    "backup": "backup",
    "regions": "regions",
    "encryption": "encryption",
}
_ANSWER = re.compile(r"^>\s?(.*)$")
_HEADER = re.compile(r"^## .*\((?P<seq>[a-z]+-\d{4})\)\s*$")

INSTRUCTIONS = """# Paraphrase sheet

Write each message the way *you* would type it to a chat assistant. One message per `>` line.

- Don't copy the description's wording. Vary it: casual, terse, a typo here and there is fine.
- Say only what the line asks for. Each message is a reply in an ongoing chat about the
  previous state, so "drop the second one" style references are fine only if unambiguous.
- Provider names must appear as written (the compiler needs to recognise them; typos in
  names are a different test).
- "Re-ask" lines: ask the same question again without changing anything (e.g. you heard the
  documents changed). Don't restate the whole question unless you would naturally.
- Please don't look at `deltaquery/domain.py` or the grounding code before or while writing.
- Leave a `>` line empty to skip it; a sequence with any empty line is left out.
"""


def _names(world: World, ents: Seq[str]) -> str:
    names = [world.name_of(e) for e in ents]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _cons(i: Intent) -> str:
    return "; ".join(
        f"{c.attribute.upper()} {'=' if c.op.value == 'eq' else '<='} {c.value_seconds} s"
        for c in i.constraints
    )


def describe_state(world: World, i: Intent) -> str:
    s = f"{_names(world, i.entities)} | {', '.join(_TOPIC[d] for d in i.dimensions)}"
    return s + (f" | requirements: {_cons(i)}" if i.constraints else "")


def describe_step(world: World, prior: Intent | None, step: Step) -> str:
    gold = Intent.model_validate(step.intent)
    if step.kind in ("new", "unrelated"):
        what = "New question" if step.kind == "new" else "Unrelated new question"
        return f"{what}: compare {describe_state(world, gold)}"
    if step.kind in ("repeat", "source_update", "revoke_source"):
        return "Re-ask: nothing changes in the question"
    parts = []
    for op in diff_intents(prior, gold):
        if op.kind is RevisionKind.REPLACE_ENTITY:
            parts.append(
                f"swap provider {world.name_of(op.old or '')} -> {world.name_of(op.new or '')}"
            )
        elif op.kind is RevisionKind.ADD_ENTITY:
            parts.append(f"add provider {world.name_of(op.new or '')}")
        elif op.kind is RevisionKind.REMOVE_ENTITY:
            parts.append(f"remove provider {world.name_of(op.old or '')}")
        elif op.kind is RevisionKind.ADD_DIMENSION:
            parts.append(f"add topic {_TOPIC[op.new or '']}")
        elif op.kind is RevisionKind.REMOVE_DIMENSION:
            parts.append(f"remove topic {_TOPIC[op.old or '']}")
        elif op.kind is RevisionKind.ADD_CONSTRAINT:
            parts.append(f"add requirement {_fmt_op(op.new or '')}")
        elif op.kind is RevisionKind.REMOVE_CONSTRAINT:
            parts.append(f"remove requirement {_fmt_op(op.old or '')}")
    return "Change: " + "; ".join(parts)


def _fmt_op(s: str) -> str:  # "rpo le 60s" -> "RPO <= 60 s"
    attr, op, val = s.split()
    return f"{attr.upper()} {'=' if op == 'eq' else '<='} {val[:-1]} s"


def select(seqs: Seq[Sequence], k: int) -> list[Sequence]:
    """Deterministic greedy pick of k sequences covering as many revision kinds as possible
    (rare kinds first), then in order."""
    total = Counter(s.kind for q in seqs for s in q.steps)
    chosen: list[Sequence] = []
    have: Counter[str] = Counter()
    pool = list(seqs)
    while pool and len(chosen) < k:

        def gain(q: Sequence) -> float:
            return sum(1 / (1 + have[s.kind]) / total[s.kind] for s in q.steps)

        best = max(pool, key=gain)  # max keeps the first on ties: stable
        pool.remove(best)
        chosen.append(best)
        have.update(s.kind for s in best.steps)
    return sorted(chosen, key=lambda q: q.id)


def build_sheet(world: World, seqs: Seq[Sequence], source: str) -> str:
    out = [INSTRUCTIONS, f"<!-- {SHEET_VERSION}; {source} -->", ""]
    for n, seq in enumerate(seqs, 1):
        out.append(f"## Conversation {n} ({seq.id})")
        out.append("")
        prior: Intent | None = None
        for i, step in enumerate(seq.steps, 1):
            if prior is not None:
                out.append(f"   _(currently: {describe_state(world, prior)})_")
            out.append(f"{i}. {describe_step(world, prior, step)}")
            out.append(">")
            out.append("")
            prior = Intent.model_validate(step.intent)
    return "\n".join(out)


def parse_sheet(text: str) -> dict[str, list[str]]:
    """seq id -> answers in step order (empty string for an unanswered line)."""
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        h = _HEADER.match(line.strip())
        if h:
            cur = out.setdefault(h.group("seq"), [])
            continue
        a = _ANSWER.match(line.strip())
        if a and cur is not None:
            cur.append(a.group(1).strip())
    return out


def apply_texts(seqs: Seq[Sequence], answers: dict[str, list[str]]) -> list[Sequence]:
    """Sequences with every step answered, texts replaced. Others are dropped."""
    out = []
    for seq in seqs:
        a = answers.get(seq.id)
        if a is None or len(a) != len(seq.steps) or not all(a):
            continue
        steps = tuple(s.model_copy(update={"text": t}) for s, t in zip(seq.steps, a, strict=True))
        out.append(seq.model_copy(update={"steps": steps}))
    return out
