"""Domain configuration: the closed vocabularies a deployment compiles questions into.

A `Domain` names the topics (retrieval dimensions) and constrainable attributes, and for
each one the deterministic pieces that the compiler, grounding, planner and claim extractor
use: keyword patterns (grounding), a retrieval query template (planner), an optional TTL, and
the label pattern a claim must follow (claim extraction).

Nothing here comes from an LLM. Changing any field changes `fingerprint`, which is folded
into the claim-check operator version, so evidence checked under an old configuration is not
reused under a new one.

Constraint values are durations in seconds for every attribute (the only value type so far).
Other units (currency, percentages) are a deliberate later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deltaquery.canonical import hash_canonical


@dataclass(frozen=True)
class TopicSpec:
    name: str
    query: str  # retrieval query template used by the planner
    keywords: tuple[str, ...]  # regexes over the casefolded message (grounding)
    ttl_s: int | None = None  # retrieval expires after this long even on an unchanged snapshot


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    query: str  # targeted retrieval query when the attribute is constrained
    keywords: tuple[str, ...]  # regexes over the casefolded message (grounding)
    claim_label: str  # regex for the label a stated value must follow (claim extraction)


@dataclass(frozen=True)
class Domain:
    name: str
    version: str
    entity_noun: str  # plural, e.g. "cloud providers"; used in the compiler prompt
    entity_short: str  # short plural, e.g. "providers"
    topics: tuple[TopicSpec, ...]
    attributes: tuple[AttributeSpec, ...]
    prompt_hints: tuple[str, ...] = ()  # extra lines for the compiler prompt
    constraint_example: str = ""
    _topics: dict[str, TopicSpec] = field(init=False, repr=False, compare=False)
    _attrs: dict[str, AttributeSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        topics = {t.name: t for t in self.topics}
        attrs = {a.name: a for a in self.attributes}
        if len(topics) != len(self.topics) or len(attrs) != len(self.attributes):
            raise ValueError(f"domain {self.name}: duplicate topic or attribute names")
        if not topics:
            raise ValueError(f"domain {self.name}: needs at least one topic")
        object.__setattr__(self, "_topics", topics)
        object.__setattr__(self, "_attrs", attrs)

    @property
    def topic_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.topics)

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attributes)

    def topic(self, name: str) -> TopicSpec:
        return self._topics[name]

    def attribute(self, name: str) -> AttributeSpec:
        return self._attrs[name]

    @property
    def claim_labels(self) -> dict[str, str]:
        return {a.name: a.claim_label for a in self.attributes}

    @property
    def fingerprint(self) -> str:
        return hash_canonical(
            "domain",
            {
                "name": self.name,
                "version": self.version,
                "topics": [vars(t) for t in self.topics],
                "attributes": [vars(a) for a in self.attributes],
            },
        )


CLOUD = Domain(
    name="cloud",
    version="1",
    entity_noun="cloud providers",
    entity_short="providers",
    topics=(
        TopicSpec(
            "disaster_recovery",
            "disaster recovery failover replication region",
            (r"disaster[\s-]*recovery", r"\bdr\b", r"\bfailover\b"),
        ),
        TopicSpec(
            "pricing",
            "pricing billed cost charges fee per gigabyte",
            (r"\bpric(?:e|es|ing)\b", r"\bcosts?\b", r"\bbilling\b"),
            ttl_s=24 * 3600,  # prices go stale outside our corpus versioning
        ),
        TopicSpec("backup", "backup vault snapshots retention", (r"\bbackups?\b",)),
        TopicSpec("regions", "regions zones availability", (r"\bregions?\b",)),
        TopicSpec("encryption", "encryption at rest keys", (r"\bencrypt(?:ion|ed|s)?\b",)),
    ),
    attributes=(
        AttributeSpec(
            "rpo",
            "recovery point objective RPO data loss replication",
            (r"\brpo\b", r"recovery[\s-]+point"),
            r"(?:\bRPO\b|recovery point objective)",
        ),
        AttributeSpec(
            "rto",
            "recovery time objective RTO downtime restore",
            (r"\brto\b", r"recovery[\s-]+time"),
            r"(?:\bRTO\b|recovery time objective)",
        ),
    ),
    prompt_hints=('"DR" means disaster_recovery.',),
    constraint_example=('"RPO of zero" -> {"attribute":"rpo","op":"eq","value_seconds":0}.'),
)
DEFAULT_DOMAIN = CLOUD
