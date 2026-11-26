"""Template planner: intent slots -> plan DAG. No LLM involvement.

Per entity, one hybrid branch (bm25 + dense -> rrf) per dimension and per constrained
attribute; an entity-level merge; then a final merge across entities. Node ids are derived
from slots, so the same slot gets the same id across revisions (useful for traces; reuse itself
is decided only by signatures).

Constraints currently *add* targeted retrieval for the constrained attribute. Checking whether
existing evidence already suffices (and skipping that retrieval) is later work; always
retrieving is the conservative choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deltaquery.compiler.intent import Dimension, Intent
from deltaquery.domain import DEFAULT_DOMAIN, Domain
from deltaquery.operators.bm25 import BM25Index, BM25Params
from deltaquery.operators.coverage import ClaimCheckParams
from deltaquery.operators.dense import DenseIndex, DenseParams
from deltaquery.operators.fusion import RRFParams
from deltaquery.runtime.models import FreshnessPolicy, Node, Plan

PLANNER_VERSION = "hybrid-template-v1"

DIMENSION_QUERIES: dict[str, str] = {t.name: t.query for t in DEFAULT_DOMAIN.topics}
# Planner v2 candidate (Phase 7, pre-registered): DR retrieval also names the recovery
# objectives, which are core disaster-recovery concepts.
DIMENSION_QUERIES_V2: dict[str, str] = {
    **DIMENSION_QUERIES,
    Dimension.DISASTER_RECOVERY: (
        "disaster recovery failover replication region "
        "recovery point objective RPO recovery time objective RTO"
    ),
}

# Pre-registered variants (project.md, Phase 7). Values are PlannerConfig overrides.
PLANNER_VARIANTS: dict[str, dict[str, object]] = {
    "A": {},
    "B": {"dimension_queries": DIMENSION_QUERIES_V2},
    "C": {"fused_k": 5},
    "D": {"dimension_queries": DIMENSION_QUERIES_V2, "fused_k": 5},
}
ATTRIBUTE_QUERIES: dict[str, str] = {a.name: a.query for a in DEFAULT_DOMAIN.attributes}
OUTPUT_ID = "answer_evidence"


@dataclass(frozen=True)
class PlannerConfig:
    bm25: BM25Index
    dense: DenseIndex
    access_scope: str
    candidates_k: int = 5
    fused_k: int = 3
    # Topic queries, attribute queries and TTLs default to the domain's; the two overrides
    # exist for planner variants.
    domain: Domain = DEFAULT_DOMAIN
    dimension_queries: Mapping[str, str] | None = None
    rrf_k: int = 60
    # Retrieval for these dimensions expires after N seconds even on an unchanged snapshot
    # (e.g. prices go stale outside our corpus versioning). None = the domain's TTLs.
    ttl_s: Mapping[str, int] | None = None
    # Evidence sufficiency: a constrained attribute gets a `claim_check` node over the entity's
    # topic branches, and its targeted retrieval branch is skipped when the check finds an
    # exact claim. False = always run the targeted retrieval branch.
    sufficiency: bool = True


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


class Planner:
    def __init__(self, config: PlannerConfig) -> None:
        self.c = config
        d = config.domain
        self._topic_q = dict(config.dimension_queries or {t.name: t.query for t in d.topics})
        self._attr_q = {a.name: a.query for a in d.attributes}
        self._ttl = dict(
            config.ttl_s
            if config.ttl_s is not None
            else {t.name: t.ttl_s for t in d.topics if t.ttl_s is not None}
        )

    def _retrieval(
        self,
        base: str,
        entity: str,
        query: str,
        max_age_s: int | None = None,
        gate: str | None = None,
    ) -> list[Node]:
        c = self.c
        common: dict[str, Any] = {
            "access_scope": c.access_scope,
            "output_contract": "ranked_passages",
            "skip_if_nonempty": gate,
        }
        fresh: dict[str, Any] = (
            {"freshness_policy": FreshnessPolicy.TTL, "max_age_s": max_age_s}
            if max_age_s is not None
            else {}
        )
        lex = Node(
            id=f"{base}__lex",
            operator="bm25",
            canonical_parameters=BM25Params(
                query=query, top_k=c.candidates_k, entity=entity
            ).model_dump(mode="json"),
            source_snapshot=c.bm25.snapshot_id,
            index_version=c.bm25.version,
            model_version=None,
            **common,
            **fresh,
        )
        dense = Node(
            id=f"{base}__dense",
            operator="dense",
            canonical_parameters=DenseParams(
                query=query, top_k=c.candidates_k, entity=entity
            ).model_dump(mode="json"),
            source_snapshot=c.dense.snapshot_id,
            index_version=c.dense.version,
            model_version=c.dense.model_version,
            **common,
            **fresh,
        )
        fused = Node(
            id=base,
            operator="rrf",
            canonical_parameters=RRFParams(k=c.rrf_k, top_k=c.fused_k).model_dump(mode="json"),
            input_ids=(lex.id, dense.id),
            source_snapshot=None,
            index_version=None,
            model_version=None,
            **common,
        )
        return [lex, dense, fused]

    def _claim_check(self, node_id: str, attr: str, inputs: list[str]) -> Node:
        c = self.c
        return Node(
            id=node_id,
            operator="claim_check",
            canonical_parameters=ClaimCheckParams(attribute=attr).model_dump(mode="json"),
            input_ids=tuple(inputs),
            source_snapshot=c.bm25.passages.source_snapshot,
            index_version=c.bm25.passages.version,
            model_version=None,
            access_scope=c.access_scope,
            output_contract="ranked_passages",
        )

    def _merge(self, node_id: str, inputs: list[str]) -> Node:
        return Node(
            id=node_id,
            operator="merge",
            canonical_parameters={},
            input_ids=tuple(inputs),
            source_snapshot=None,
            index_version=None,
            model_version=None,
            access_scope=self.c.access_scope,
            output_contract="evidence_set",
        )

    def build(self, intent: Intent) -> Plan:
        nodes: list[Node] = []
        entity_outputs: list[str] = []
        attributes = sorted({c.attribute for c in intent.constraints})
        for entity in intent.entities:  # canonical (sorted) order
            e = _slug(entity)
            branches: list[str] = []
            for dim in intent.dimensions:
                base = f"{e}__{dim}"
                nodes += self._retrieval(base, entity, self._topic_q[dim], self._ttl.get(dim))
                branches.append(base)
            topic_branches = list(branches)
            for attr in attributes:
                base = f"{e}__{attr}"
                gate = None
                if self.c.sufficiency:
                    gate = f"{base}__check"
                    nodes.append(self._claim_check(gate, attr, topic_branches))
                    branches.append(gate)
                # Targeted retrieval; skipped when the claim check already found the value.
                nodes += self._retrieval(base, entity, self._attr_q[attr], gate=gate)
                branches.append(base)
            nodes.append(self._merge(f"{e}__evidence", branches))
            entity_outputs.append(f"{e}__evidence")
        nodes.append(self._merge(OUTPUT_ID, entity_outputs))
        return Plan(nodes=tuple(nodes), output_id=OUTPUT_ID)
