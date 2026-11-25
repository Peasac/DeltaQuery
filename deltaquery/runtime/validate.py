"""Conservative reuse validation.

A signature hit only *proposes* reuse. This check re-verifies every field the artifact was
computed under against the current node, operator, inputs, access policy and clock. Any
mismatch -> re-execute. Several checks are implied by the signature already; they are repeated
on purpose (defense in depth against hashing or storage bugs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from deltaquery.evidence.models import Artifact
from deltaquery.operators.base import Operator
from deltaquery.runtime.models import FreshnessPolicy, Node


@dataclass(frozen=True)
class ReuseContext:
    signature: str
    inputs: Sequence[Artifact]
    access_policy_version: str
    revoked_sources: frozenset[str]
    principal_may_read: bool
    now: datetime
    last_verified_at: datetime | None  # only needed for TTL nodes


def reuse_rejection_reason(
    node: Node, operator: Operator[Any], cached: Artifact, ctx: ReuseContext
) -> str | None:
    """Return None if `cached` may be reused for `node`, else the first failing check."""
    bound = operator.bound_versions
    checks: list[tuple[str, bool]] = [
        # Access first: fail closed before anything else is considered.
        ("principal_scope", ctx.principal_may_read),
        ("access_scope", cached.access_scope == node.access_scope),
        ("access_policy_version", cached.access_policy_version == ctx.access_policy_version),
        ("revoked_source", not any(e.source_id in ctx.revoked_sources for e in cached.evidence)),
        ("signature", cached.signature == ctx.signature),
        ("operator", cached.operator == node.operator == operator.name),
        ("operator_version", cached.operator_version == operator.version),
        ("output_contract", cached.output_contract == node.output_contract),
        ("source_snapshot", cached.source_snapshot == node.source_snapshot),
        ("source_snapshot_current", cached.source_snapshot == bound.source_snapshot),
        ("index_version", cached.index_version == node.index_version == bound.index_version),
        ("model_version", cached.model_version == node.model_version == bound.model_version),
        ("inputs", cached.input_signatures == tuple(a.signature for a in ctx.inputs)),
        ("freshness", _fresh(node, ctx)),
    ]
    for name, ok in checks:
        if not ok:
            return name
    return None


def _fresh(node: Node, ctx: ReuseContext) -> bool:
    if node.freshness_policy is FreshnessPolicy.SNAPSHOT:
        return True  # snapshot equality is checked above
    if node.freshness_policy is FreshnessPolicy.TTL and node.max_age_s is not None:
        if ctx.last_verified_at is None:
            return False  # unknown freshness -> recompute
        return ctx.now - ctx.last_verified_at <= timedelta(seconds=node.max_age_s)
    return False  # unknown policy -> recompute
