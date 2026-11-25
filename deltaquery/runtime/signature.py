"""Content-addressed node signatures.

signature = H(operator, operator version, canonical params, ordered input artifact hashes,
              source snapshot, index version, model version, access scope,
              access-policy version, freshness policy + max age, output contract)

The plan-local node id is deliberately excluded, so the same computation is recognized across
plans. Inputs are kept in declared order (operators may be order-sensitive).
"""

from __future__ import annotations

from collections.abc import Sequence

from deltaquery.canonical import hash_canonical
from deltaquery.runtime.models import Node


def node_signature(
    node: Node,
    operator_version: str,
    input_artifact_hashes: Sequence[str],
    access_policy_version: str,
    gate: str | None = None,
) -> str:
    return hash_canonical(
        "node",
        {
            "operator": node.operator,
            "operator_version": operator_version,
            "params": node.canonical_parameters,
            "inputs": list(input_artifact_hashes),
            "source_snapshot": node.source_snapshot,
            "index_version": node.index_version,
            "model_version": node.model_version,
            "access_scope": node.access_scope,
            "access_policy_version": access_policy_version,
            "freshness_policy": node.freshness_policy.value,
            "max_age_s": node.max_age_s,
            "output_contract": node.output_contract,
            # For gated nodes: the gate's decision ("run" | "skip"), not its content.
            "gate": gate,
        },
    )
