"""Reciprocal rank fusion over explicit candidate-list inputs.

Fusion is its own node: it depends on every candidate list through declared input edges, so a
change to fusion parameters recomputes only fusion (and its dependents), while a change to one
retriever recomputes that retriever and fusion.

Inputs must all be `ranked_passages` in the shared span format; a passage that appears with a
different source_version or content_hash in two inputs is a span-format violation and fails.
RRF score for a passage = sum over inputs (declared order) of 1 / (k + rank), rank from 1.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from deltaquery.evidence.models import Artifact, CoverageType, Evidence
from deltaquery.operators.base import RANKED_PASSAGES, BoundVersions, ExecutionContext, Operator


class FusionInputError(ValueError):
    pass


class RRFParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=60, ge=1)
    top_k: int = Field(ge=1, le=100)


SpanKey = tuple[str, int, int]


class RRFOperator(Operator[RRFParams]):
    name = "rrf"
    version = "1"
    output_contract = RANKED_PASSAGES
    params_model = RRFParams
    min_inputs = 2
    max_inputs = None

    @property
    def bound_versions(self) -> BoundVersions:
        return BoundVersions(source_snapshot=None, index_version=None, model_version=None)

    def access_dependencies(
        self, params: RRFParams, access_scope: str, revoked: frozenset[str]
    ) -> frozenset[str]:
        return frozenset()  # depends on revocations only through its inputs' hashes

    async def run(
        self, params: RRFParams, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        first: dict[SpanKey, Evidence] = {}
        scores: dict[SpanKey, float] = {}
        for art in inputs:
            if art.output_contract != RANKED_PASSAGES:
                raise FusionInputError(f"input {art.signature[:12]} is not ranked_passages")
            for rank, e in enumerate(art.evidence, start=1):
                key = (e.source_id, e.span.start, e.span.end)
                seen = first.setdefault(key, e)
                if (seen.source_version, seen.content_hash) != (e.source_version, e.content_hash):
                    raise FusionInputError(f"span format mismatch for {key}")
                scores[key] = scores.get(key, 0.0) + 1.0 / (params.k + rank)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        return [
            first[key].model_copy(
                update={
                    "score": scores[key],
                    "retrieval_node_id": ctx.signature,
                    "coverage_type": CoverageType.FUSED,
                    "extracted_claims": (),
                    "created_at": ctx.now,
                }
            )
            for key in ordered[: params.top_k]
        ]
