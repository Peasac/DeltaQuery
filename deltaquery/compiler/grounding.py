"""Deterministic grounding of LLM-proposed intent changes against the user's own words.

The LLM proposes a complete intent; this module approves each *change* relative to the prior
intent only if the message names it:

- entity added or removed      -> the message names that provider
- dimension added or removed   -> the message uses a word for that topic
- constraint added             -> the message names the attribute and states the value
- constraint removed           -> the message names the attribute

A message that names two or more providers, none of them in the prior intent, is a new
question: the prior is ignored and the proposal is grounded as a fresh question.
A fresh question must name at least one provider (otherwise CompileError upstream).
`task` is derived from the entity count (compare iff >= 2), which is what it means.

Ungrounded changes are dropped, not retried (at temperature 0 a retry repeats the output).
The keyword lists are deliberately small; a missed synonym makes the compiler keep the prior
slot (a wrong plan, never a stale reuse), and such misses should be added as fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from deltaquery.compiler.intent import Attribute, Constraint, Dimension, Intent, Task

GROUNDING_VERSION = "grounding-v1"

DIMENSION_WORDS: dict[Dimension, tuple[str, ...]] = {
    Dimension.DISASTER_RECOVERY: (r"disaster[\s-]*recovery", r"\bdr\b", r"\bfailover\b"),
    Dimension.PRICING: (r"\bpric(?:e|es|ing)\b", r"\bcosts?\b", r"\bbilling\b"),
    Dimension.BACKUP: (r"\bbackups?\b",),
    Dimension.REGIONS: (r"\bregions?\b",),
    Dimension.ENCRYPTION: (r"\bencrypt(?:ion|ed|s)?\b",),
}
ATTRIBUTE_WORDS: dict[Attribute, tuple[str, ...]] = {
    Attribute.RPO: (r"\brpo\b", r"recovery[\s-]+point"),
    Attribute.RTO: (r"\brto\b", r"recovery[\s-]+time"),
}
_NUMBER_WORDS = {0: ("zero", "no")}


@dataclass(frozen=True)
class Mentions:
    entities: frozenset[str]
    dimensions: frozenset[Dimension]
    attributes: frozenset[Attribute]

    @property
    def empty(self) -> bool:
        return not (self.entities or self.dimensions or self.attributes)


def _has(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _entity_pattern(entity_id: str) -> str:
    # Entity ids are normalized with "-" for spaces; accept either in the text.
    body = r"[-\s]+".join(re.escape(part) for part in entity_id.split("-"))
    return r"\b" + body + r"\b"


def mentions(text: str, vocabulary: Iterable[str]) -> Mentions:
    t = text.casefold()
    ents = frozenset(e for e in vocabulary if re.search(_entity_pattern(e), t))
    dims = frozenset(d for d, pats in DIMENSION_WORDS.items() if _has(t, pats))
    attrs = frozenset(a for a, pats in ATTRIBUTE_WORDS.items() if _has(t, pats))
    return Mentions(ents, dims, attrs)


def _value_stated(text: str, seconds: int) -> bool:
    t = text.casefold()
    words = _NUMBER_WORDS.get(seconds, ())
    if any(re.search(rf"\b{w}\b", t) for w in words):
        return True
    if re.search(rf"\b{seconds}\s*(?:s|sec|secs|seconds?)\b", t):
        return True
    if seconds % 60 == 0 and re.search(rf"\b{seconds // 60}\s*(?:min|mins|minutes?)\b", t):
        return True
    return seconds % 3600 == 0 and bool(re.search(rf"\b{seconds // 3600}\s*(?:h|hours?)\b", t))


def _grounded_new_constraints(
    proposed: Iterable[Constraint], m: Mentions, text: str
) -> list[Constraint]:
    return [
        c for c in proposed if c.attribute in m.attributes and _value_stated(text, c.value_seconds)
    ]


@dataclass(frozen=True)
class Grounded:
    entities: tuple[str, ...]
    dimensions: tuple[Dimension, ...]
    constraints: tuple[Constraint, ...]
    notes: tuple[str, ...]  # one line per dropped/restored change


def _ground_fresh(text: str, proposed: Intent, m: Mentions) -> Grounded:
    notes: list[str] = []
    ents = [e for e in proposed.entities if e in m.entities]
    notes += [f"dropped unnamed entity {e}" for e in proposed.entities if e not in m.entities]
    if m.dimensions:
        dims = [d for d in proposed.dimensions if d in m.dimensions]
        notes += [
            f"dropped unnamed dimension {d.value}"
            for d in proposed.dimensions
            if d not in m.dimensions
        ]
        if not dims:  # model picked none of the named topics: use the named ones
            dims = sorted(m.dimensions, key=lambda d: d.value)
            notes.append("dimensions taken from the message")
    else:
        dims = list(proposed.dimensions)  # nothing to ground against
    cons = _grounded_new_constraints(proposed.constraints, m, text)
    notes += [
        f"dropped ungrounded constraint {c.attribute.value}"
        for c in proposed.constraints
        if c not in cons
    ]
    return Grounded(tuple(ents), tuple(dims), tuple(cons), tuple(notes))


def ground(text: str, prior: Intent | None, proposed: Intent, m: Mentions) -> Grounded:
    if prior is None or (
        len(m.entities - set(prior.entities)) >= 2 and not (m.entities & set(prior.entities))
    ):
        return _ground_fresh(text, proposed, m)  # new question: the prior is ignored
    notes: list[str] = []
    p_e, n_e = set(prior.entities), set(proposed.entities)
    ents = set(p_e)
    for e in sorted(n_e - p_e):
        if e in m.entities:
            ents.add(e)
        else:
            notes.append(f"dropped unnamed entity {e}")
    for e in sorted(p_e - n_e):
        if e in m.entities:
            ents.discard(e)
        else:
            notes.append(f"kept entity {e} (removal not requested)")

    p_d, n_d = set(prior.dimensions), set(proposed.dimensions)
    dims = set(p_d)
    for d in sorted(n_d - p_d, key=lambda d: d.value):
        if d in m.dimensions:
            dims.add(d)
        else:
            notes.append(f"dropped unnamed dimension {d.value}")
    for d in sorted(p_d - n_d, key=lambda d: d.value):
        if d in m.dimensions:
            dims.discard(d)
        else:
            notes.append(f"kept dimension {d.value} (removal not requested)")
    if not dims:
        dims = set(p_d)
        notes.append("kept prior dimensions (none would remain)")

    p_c, n_c = set(prior.constraints), set(proposed.constraints)
    cons = set(p_c)
    for c in _grounded_new_constraints(sorted(n_c - p_c, key=Constraint.sort_key), m, text):
        cons = {x for x in cons if x.attribute != c.attribute} | {c}  # one value per attribute
    notes += [
        f"dropped ungrounded constraint {c.attribute.value}"
        for c in sorted(n_c - p_c, key=Constraint.sort_key)
        if c not in cons
    ]
    for c in sorted(p_c - n_c, key=Constraint.sort_key):
        if c not in cons:
            continue  # already replaced by a grounded new value for the same attribute
        if c.attribute in m.attributes:
            cons.discard(c)
        else:
            notes.append(f"kept constraint {c.attribute.value} (removal not requested)")
    return Grounded(
        tuple(sorted(ents)),
        tuple(sorted(dims, key=lambda d: d.value)),
        tuple(sorted(cons, key=Constraint.sort_key)),
        tuple(notes),
    )


def task_for(entities: Iterable[str]) -> Task:
    return Task.COMPARE if len(list(entities)) >= 2 else Task.DESCRIBE
