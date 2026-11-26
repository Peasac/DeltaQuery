"""Paraphrase sheet: the sheet hides the benchmark wording and parses back losslessly."""

from __future__ import annotations

from deltaquery.eval.paraphrase import apply_texts, build_sheet, parse_sheet, select
from deltaquery.eval.synth import build_world, generate_sequences


def test_sheet_round_trip_and_hides_template_wording() -> None:
    world = build_world(7)
    seqs = generate_sequences(world, 12, 29)
    chosen = select(seqs, 4)
    sheet = build_sheet(world, chosen, "test")
    for q in chosen:
        for s in q.steps:
            if s.kind not in ("new", "unrelated"):
                assert s.text not in sheet  # revision templates never shown
    filled_lines = []
    n = 0
    for line in sheet.splitlines():
        if line.strip() == ">":
            n += 1
            line = f"> message {n}"
        filled_lines.append(line)
    parsed = parse_sheet("\n".join(filled_lines))
    assert list(parsed) == [q.id for q in chosen]
    out = {q.id: q for q in apply_texts(seqs, parsed)}
    assert sorted(out) == [q.id for q in chosen]
    first = out[chosen[0].id]
    assert first.steps[0].text == "message 1"
    assert [s.intent for s in first.steps] == [s.intent for s in chosen[0].steps]


def test_partially_answered_conversation_is_left_out() -> None:
    world = build_world(7)
    seqs = generate_sequences(world, 6, 29)
    sheet = build_sheet(world, select(seqs, 2), "test")
    first = True
    lines = []
    for line in sheet.splitlines():
        if line.strip() == ">":
            line = ">" if first else "> x"
            first = False
        lines.append(line)
    assert len(apply_texts(seqs, parse_sheet("\n".join(lines)))) == 1
