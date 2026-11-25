"""Deterministic join of candidate lists (stand-in for the compare/join node in Phase 1).

This is *not* rank fusion (Phase 2): it concatenates inputs in declared order, keeps the first
occurrence of each (source_id, span), and passes evidence through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from deltaquery.evidence.models import Artifact, Evidence
from deltaquery.operators.base import BoundVersions, ExecutionContext, Operator


class MergeParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MergeOperator(Operator[MergeParams]):
    name = "merge"
    version = "1"
    output_contract = "evidence_set"
    params_model = MergeParams
    min_inputs = 1
    max_inputs = None

    @property
    def bound_versions(self) -> BoundVersions:
        # Depends on sources only through its inputs, whose hashes are in the signature.
        return BoundVersions(source_snapshot=None, index_version=None, model_version=None)

    def access_dependencies(
        self, params: MergeParams, access_scope: str, revoked: frozenset[str]
    ) -> frozenset[str]:
        return frozenset()  # depends on revocations only through its inputs' hashes

    async def run(
        self, params: MergeParams, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        seen: set[tuple[str, int, int]] = set()
        out: list[Evidence] = []
        for art in inputs:
            for e in art.evidence:
                key = (e.source_id, e.span.start, e.span.end)
                if key not in seen:
                    seen.add(key)
                    out.append(e)
        return out
