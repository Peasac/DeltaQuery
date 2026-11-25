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
from dataclasses import dataclass, field
from typing import Any

from deltaquery.compiler.intent import Attribute, Dimension, Intent
from deltaquery.operators.bm25 import BM25Index, BM25Params
from deltaquery.operators.coverage import ClaimCheckParams
from deltaquery.operators.dense import DenseIndex, DenseParams
from deltaquery.operators.fusion import RRFParams
from deltaquery.runtime.models import FreshnessPolicy, Node, Plan

PLANNER_VERSION = "hybrid-template-v1"

DIMENSION_QUERIES: dict[Dimension, str] = {
    Dimension.DISASTER_RECOVERY: "disaster recovery failover replication region",
    Dimension.PRICING: "pricing billed cost charges fee per gigabyte",
    Dimension.BACKUP: "backup vault snapshots retention",
    Dimension.REGIONS: "regions zones availability",
    Dimension.ENCRYPTION: "encryption at rest keys",
}
ATTRIBUTE_QUERIES: dict[Attribute, str] = {
    Attribute.RPO: "recovery point objective RPO data loss replication",
    Attribute.RTO: "recovery time objective RTO downtime restore",
}
OUTPUT_ID = "answer_evidence"


@dataclass(frozen=True)
class PlannerConfig:
    bm25: BM25Index
    dense: DenseIndex
    access_scope: str
    candidates_k: int = 5
    fused_k: int = 3
    rrf_k: int = 60
    # Retrieval for these dimensions expires after N seconds even on an unchanged snapshot
    # (prices go stale outside our corpus versioning).
    ttl_s: Mapping[Dimension, int] = field(default_factory=lambda: {Dimension.PRICING: 24 * 3600})
    # Evidence sufficiency: a constrained attribute gets a `claim_check` node over the entity's
    # topic branches, and its targeted retrieval branch is skipped when the check finds an
    # exact claim. False = always run the targeted retrieval branch.
    sufficiency: bool = True


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


class Planner:
    def __init__(self, config: PlannerConfig) -> None:
        self.c = config

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

    def _claim_check(self, node_id: str, attr: Attribute, inputs: list[str]) -> Node:
        c = self.c
        return Node(
            id=node_id,
            operator="claim_check",
            canonical_parameters=ClaimCheckParams(attribute=attr.value).model_dump(mode="json"),
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
        attributes = sorted({c.attribute for c in intent.constraints}, key=lambda a: a.value)
        for entity in intent.entities:  # canonical (sorted) order
            e = _slug(entity)
            branches: list[str] = []
            for dim in intent.dimensions:
                base = f"{e}__{dim.value}"
                nodes += self._retrieval(
                    base, entity, DIMENSION_QUERIES[dim], self.c.ttl_s.get(dim)
                )
                branches.append(base)
            topic_branches = list(branches)
            for attr in attributes:
                base = f"{e}__{attr.value}"
                gate = None
                if self.c.sufficiency:
                    gate = f"{base}__check"
                    nodes.append(self._claim_check(gate, attr, topic_branches))
                    branches.append(gate)
                # Targeted retrieval; skipped when the claim check already found the value.
                nodes += self._retrieval(base, entity, ATTRIBUTE_QUERIES[attr], gate=gate)
                branches.append(base)
            nodes.append(self._merge(f"{e}__evidence", branches))
            entity_outputs.append(f"{e}__evidence")
        nodes.append(self._merge(OUTPUT_ID, entity_outputs))
        return Plan(nodes=tuple(nodes), output_id=OUTPUT_ID)
