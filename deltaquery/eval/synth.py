"""DeltaBench-synth: a generated world (corpus v0/v1) and revision sequences with gold labels.

Everything is deterministic from the seed. Provider names are invented from syllables. Each
document has three paragraphs; the *fact paragraphs* carry exact values and are the gold
evidence for their (entity, dimension) slot. Snapshot v1 changes the DR and pricing facts of a
subset of providers, so a source update makes previously retrieved evidence stale.

This is synthetic data with exact labels. It validates the harness and the reuse/correctness
comparison; it says little about retrieval quality on real documents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from deltaquery.compiler.intent import Attribute, Comparison, Dimension, Intent, Task
from deltaquery.evidence.corpus import Corpus, SourceDocument

DATASET_VERSION = "deltabench-synth-v1"
SCOPE = "bench"
N_ENTITIES = 40

_A = ["ar", "bel", "cor", "del", "es", "fen", "gal", "hal", "is", "jor", "kel", "lu", "mor",
      "nev", "or", "pel", "quin", "ros", "sel", "tav", "ul", "ver", "wyn", "zel"]  # fmt: skip
_B = ["vane", "drick", "mora", "tris", "vell", "dor", "nix", "wick", "thas", "bryn"]

DIM_WORDS = {
    Dimension.DISASTER_RECOVERY: "disaster recovery",
    Dimension.PRICING: "pricing",
    Dimension.BACKUP: "backup",
    Dimension.REGIONS: "regions",
    Dimension.ENCRYPTION: "encryption",
}
# Paragraph indexes holding gold facts for each slot.
GOLD_PARAGRAPHS: dict[Dimension, tuple[int, ...]] = {
    Dimension.DISASTER_RECOVERY: (1, 2),
    Dimension.PRICING: (1,),
    Dimension.BACKUP: (1,),
    Dimension.REGIONS: (1,),
    Dimension.ENCRYPTION: (1,),
}
ATTRIBUTE_PARAGRAPH: dict[Attribute, tuple[Dimension, int]] = {
    Attribute.RPO: (Dimension.DISASTER_RECOVERY, 1),
    Attribute.RTO: (Dimension.DISASTER_RECOVERY, 2),
}


def entity_names(seed: int) -> list[str]:
    rng = random.Random(seed)
    combos = [a + b for a in _A for b in _B]
    rng.shuffle(combos)
    return [c.capitalize() for c in combos[:N_ENTITIES]]


def source_id(entity: str, dim: Dimension) -> str:
    return f"{entity.lower()}-{dim.value.replace('_', '-')}"


@dataclass(frozen=True)
class Facts:
    rpo: int
    rto: int
    price: str  # dollars per GB, as written in the text
    retention_days: int
    regions: int
    key_mode: str


def _facts(rng: random.Random) -> Facts:
    return Facts(
        rpo=rng.choice([0, 5, 15, 30, 60, 300]),
        rto=rng.choice([60, 300, 900, 1800, 3600]),
        price=f"{rng.randint(1, 9) / 100:.2f}",
        retention_days=rng.choice([7, 14, 30, 90]),
        regions=rng.randint(3, 12),
        key_mode=rng.choice(["customer-managed", "provider-managed"]),
    )


def _paragraphs(name: str, dim: Dimension, f: Facts) -> list[str]:
    if dim is Dimension.DISASTER_RECOVERY:
        return [
            f"{name} disaster recovery overview. {name} replicates data to a secondary region "
            "for failover, and recovery plans start services in a defined order.",
            f"Recovery point objective (RPO): with continuous replication the {name} RPO is "
            f"{f.rpo} seconds of data loss at most.",
            f"Recovery time objective (RTO): {name} restores service within {f.rto} seconds "
            "of a regional failover.",
        ]
    if dim is Dimension.PRICING:
        return [
            f"{name} pricing overview for replication and standby capacity.",
            f"{name} bills cross-region replication at ${f.price} per gigabyte transferred; "
            "this fee is the main cost of disaster recovery.",
            "Standby compute is charged at the normal hourly rate and idle replicas cost nothing.",
        ]
    if dim is Dimension.BACKUP:
        return [
            f"{name} backup overview.",
            f"{name} backup vaults keep snapshots for {f.retention_days} days by default, "
            "with a configurable retention period.",
            "Vault lock can prevent deletion of backups before retention ends.",
        ]
    if dim is Dimension.REGIONS:
        return [
            f"{name} regions overview.",
            f"{name} operates {f.regions} regions, each with at least three availability zones.",
            "Regions are isolated geographic areas connected by low latency links.",
        ]
    return [
        f"{name} encryption overview.",
        f"{name} encrypts data at rest with {f.key_mode} keys held in a managed key service.",
        "Encryption in transit uses TLS for all service endpoints.",
    ]


class World(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    seed: int
    names: tuple[str, ...]
    facts: dict[int, dict[str, Facts]]  # snapshot -> entity id -> facts
    changed_entities: frozenset[str]  # entity ids whose DR/pricing facts differ in v1

    @property
    def entity_ids(self) -> list[str]:
        return [n.lower() for n in self.names]

    def name_of(self, entity_id: str) -> str:
        return {n.lower(): n for n in self.names}[entity_id]

    def corpus(self, snapshot: int) -> Corpus:
        docs = []
        for name in self.names:
            f = self.facts[snapshot][name.lower()]
            for dim in Dimension:
                docs.append(
                    SourceDocument(
                        source_id=source_id(name, dim),
                        entity=name.lower(),
                        title=f"{name} {DIM_WORDS[dim]}",
                        text="\n\n".join(_paragraphs(name, dim, f)),
                        access_scope=SCOPE,
                    )
                )
        return Corpus(docs)

    def fact_strings(self, snapshot: int, entity_id: str, dim: Dimension) -> list[str]:
        """Strings a correct answer about this slot should contain (for the e2e grader)."""
        f = self.facts[snapshot][entity_id]
        return {
            Dimension.DISASTER_RECOVERY: [str(f.rpo), str(f.rto)],
            Dimension.PRICING: [f.price],
            Dimension.BACKUP: [str(f.retention_days)],
            Dimension.REGIONS: [str(f.regions)],
            Dimension.ENCRYPTION: [f.key_mode],
        }[dim]


def build_world(seed: int = 7, change_fraction: float = 0.3) -> World:
    rng = random.Random(seed)
    names = entity_names(seed)
    v0 = {n.lower(): _facts(rng) for n in names}
    changed = sorted(rng.sample([n.lower() for n in names], int(len(names) * change_fraction)))
    v1 = dict(v0)
    for e in changed:
        old = v0[e]
        new = _facts(rng)
        # Guarantee the DR and pricing documents really change.
        rpo = new.rpo if new.rpo != old.rpo else (old.rpo + 1)
        price = new.price if new.price != old.price else f"{float(old.price) + 0.01:.2f}"
        v1[e] = Facts(rpo, new.rto, price, old.retention_days, old.regions, old.key_mode)
    return World(
        seed=seed, names=tuple(names), facts={0: v0, 1: v1}, changed_entities=frozenset(changed)
    )


# -- sequences ----------------------------------------------------------------------------

STEP_KINDS = (
    "new", "replace_entity", "add_entity", "remove_entity", "add_dimension",
    "remove_dimension", "add_constraint", "remove_constraint", "repeat", "unrelated",
    "source_update", "revoke_source",
)  # fmt: skip


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    text: str
    intent: dict[str, Any]  # gold intent (canonical JSON)
    event: dict[str, Any] | None = None  # {"snapshot": 1} or {"revoke": source_id}


class Sequence(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    split: str
    steps: tuple[Step, ...]


def _intent(
    task: Task, entities: list[str], dims: list[Dimension], cons: list[dict[str, Any]]
) -> Intent:
    return Intent.model_validate(
        {"task": task, "entities": entities, "dimensions": dims, "constraints": cons}
    )


def _dims_text(dims: list[Dimension]) -> str:
    words = [DIM_WORDS[d] for d in dims]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def generate_sequences(world: World, n: int = 300, seed: int = 11) -> list[Sequence]:
    rng = random.Random(seed)
    ids = world.entity_ids
    pools = {"dev": ids[: N_ENTITIES // 2], "test": ids[N_ENTITIES // 2 :]}
    dim_weights = [0.4, 0.3, 0.1, 0.1, 0.1]
    out: list[Sequence] = []

    def fresh(pool: list[str], exclude: frozenset[str] = frozenset()) -> Intent:
        ents = rng.sample([e for e in pool if e not in exclude], 2)
        dim = rng.choices(list(Dimension), dim_weights)[0]
        return _intent(Task.COMPARE, ents, [dim], [])

    def ask_text(it: Intent) -> str:
        names = [world.name_of(e) for e in it.entities]
        return f"Compare {_join(names)} on {_dims_text(list(it.dimensions))}"

    for i in range(n):
        split = "dev" if i % 2 == 0 else "test"
        pool = pools[split]
        cur = fresh(pool)
        steps = [Step(kind="new", text=ask_text(cur), intent=cur.model_dump(mode="json"))]
        snapshot_used = revoke_used = False
        length = rng.choice([4, 5, 6])
        while len(steps) < length:
            ents, dims, cons = list(cur.entities), list(cur.dimensions), list(cur.constraints)
            options = ["replace_entity", "add_dimension", "add_constraint", "repeat", "unrelated"]
            if len(ents) < 4:
                options.append("add_entity")
            if len(ents) > 2:
                options.append("remove_entity")
            if len(dims) > 1:
                options.append("remove_dimension")
            if cons:
                options.append("remove_constraint")
            if not snapshot_used:
                options += ["source_update"] * 2
            if not revoke_used:
                options += ["revoke_source"] * 2
            kind = rng.choice(options)
            event: dict[str, Any] | None = None
            if kind == "replace_entity":
                old = rng.choice(ents)
                new = rng.choice([e for e in pool if e not in ents])
                ents = [new if e == old else e for e in ents]
                text = f"Replace {world.name_of(old)} with {world.name_of(new)}"
            elif kind == "add_entity":
                new = rng.choice([e for e in pool if e not in ents])
                ents.append(new)
                text = f"Also include {world.name_of(new)}"
            elif kind == "remove_entity":
                old = rng.choice(ents)
                ents.remove(old)
                text = f"Drop {world.name_of(old)}"
            elif kind == "add_dimension":
                choices = [d for d in Dimension if d not in dims]
                if not choices:
                    continue
                d = rng.choice(choices)
                dims.append(d)
                text = f"Also compare {DIM_WORDS[d]}"
            elif kind == "remove_dimension":
                d = rng.choice(dims)
                dims.remove(d)
                text = f"Ignore {DIM_WORDS[d]} now"
            elif kind == "add_constraint":
                have = {c.attribute for c in cons}
                attrs = [a for a in Attribute if a not in have]
                if not attrs:
                    continue
                a = rng.choice(attrs)
                c: dict[str, Any]
                if a is Attribute.RPO and rng.random() < 0.5:
                    c = {"attribute": "rpo", "op": "eq", "value_seconds": 0}
                    text = "Only consider options with an RPO of zero"
                else:
                    v = rng.choice([60, 300, 900]) if a is Attribute.RTO else rng.choice([30, 60])
                    c = {"attribute": a.value, "op": "le", "value_seconds": v}
                    text = f"Only consider options with an {a.value.upper()} of at most {v} seconds"
                cons = [*(x.model_dump(mode="json") for x in cons), c]  # type: ignore[list-item]
            elif kind == "remove_constraint":
                c0 = rng.choice(cons)
                cons = [x for x in cons if x != c0]
                text = f"Remove the {c0.attribute.value.upper()} requirement"
            elif kind == "repeat":
                text = "Show that again"
            elif kind == "unrelated":
                cur = fresh(pool, exclude=frozenset(ents))
                steps.append(
                    Step(kind=kind, text=ask_text(cur), intent=cur.model_dump(mode="json"))
                )
                continue
            elif kind == "source_update":
                snapshot_used = True
                event = {"snapshot": 1}
                text = "Check again with the latest documents"
            else:  # revoke_source: a document the current question depends on
                revoke_used = True
                e = rng.choice(ents)
                d = rng.choice(dims)
                event = {"revoke": source_id(world.name_of(e), d)}
                text = "Check again with the latest documents"
            cur = _intent(
                cur.task,
                ents,
                dims,
                [x if isinstance(x, dict) else x.model_dump(mode="json") for x in cons],
            )
            steps.append(
                Step(kind=kind, text=text, intent=cur.model_dump(mode="json"), event=event)
            )
        out.append(Sequence(id=f"{split}-{i:04d}", split=split, steps=tuple(steps)))
    return out


def gold_keys(
    world: World,
    corpus: Corpus,
    intent: Intent,
    revoked: frozenset[str],
) -> set[tuple[str, int, int]]:
    """Gold evidence (source_id, span) for an intent under a snapshot and revocations."""
    from deltaquery.evidence.corpus import chunk_paragraphs

    docs = {d.source_id: d for d in corpus.documents}
    wanted: set[tuple[str, int]] = set()
    for e in intent.entities:
        name = world.name_of(e)
        for dim in intent.dimensions:
            for p in GOLD_PARAGRAPHS[dim]:
                wanted.add((source_id(name, dim), p))
        for c in intent.constraints:
            dim, p = ATTRIBUTE_PARAGRAPH[c.attribute]
            wanted.add((source_id(name, dim), p))
    out: set[tuple[str, int, int]] = set()
    for sid, idx in wanted:
        if sid in revoked:
            continue
        passages = chunk_paragraphs(docs[sid])
        out.add((sid, passages[idx].span.start, passages[idx].span.end))
    return out


__all__ = [
    "DATASET_VERSION",
    "SCOPE",
    "STEP_KINDS",
    "Comparison",
    "Sequence",
    "Step",
    "World",
    "build_world",
    "generate_sequences",
    "gold_keys",
]
