"""Conservative reuse validation.

A signature hit only *proposes* reuse. This check re-verifies every field the artifact was
computed under against the current node, operator and inputs. Any mismatch -> re-execute.
Freshness beyond snapshot equality and permission re-authorization are Phase 4; until then
the scope check is exact equality (fail closed).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deltaquery.evidence.models import Artifact
from deltaquery.operators.base import Operator
from deltaquery.runtime.models import FreshnessPolicy, Node


def reuse_rejection_reason(
    node: Node,
    operator: Operator[Any],
    signature: str,
    inputs: Sequence[Artifact],
    cached: Artifact,
) -> str | None:
    """Return None if `cached` may be reused for `node`, else a reason string."""
    bound = operator.bound_versions
    checks: list[tuple[str, bool]] = [
        ("signature", cached.signature == signature),
        ("operator", cached.operator == node.operator == operator.name),
        ("operator_version", cached.operator_version == operator.version),
        ("output_contract", cached.output_contract == node.output_contract),
        ("access_scope", cached.access_scope == node.access_scope),
        ("source_snapshot", cached.source_snapshot == node.source_snapshot),
        ("source_snapshot_current", cached.source_snapshot == bound.source_snapshot),
        ("index_version", cached.index_version == node.index_version == bound.index_version),
        ("model_version", cached.model_version == node.model_version == bound.model_version),
        ("inputs", cached.input_signatures == tuple(a.signature for a in inputs)),
        ("freshness_policy", node.freshness_policy is FreshnessPolicy.SNAPSHOT),
    ]
    for name, ok in checks:
        if not ok:
            return name
    return None
