"""Evidence-sufficiency check: does existing evidence already state the constrained value?

`claim_check(attribute)` takes an entity's topic-branch outputs as inputs and returns the
evidence that states an exact claim for `attribute` (operators/claims.py), with the claims
attached, or nothing. It never retrieves.

The planner gates the targeted retrieval branch for that attribute on this node
(`Node.skip_if_nonempty`): if the check returns evidence, the retrieval branch is skipped;
otherwise it runs. The gate's *decision* (not its content) is part of the gated nodes'
signatures, so a retrieval branch that keeps being needed across revisions is reused, not
re-run.

Passage text is resolved from the bound passage set and checked against the evidence content
hash; evidence that cannot be verified does not count as covering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from deltaquery.evidence.corpus import Passage, PassageSet
from deltaquery.evidence.models import Artifact, CoverageType, Evidence
from deltaquery.operators.base import RANKED_PASSAGES, BoundVersions, ExecutionContext, Operator
from deltaquery.operators.claims import EXTRACTOR_VERSION, extract_claims


class ClaimCheckParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attribute: Literal["rpo", "rto"]


class ClaimCheckOperator(Operator[ClaimCheckParams]):
    name = "claim_check"
    version = f"1+{EXTRACTOR_VERSION}"
    output_contract = RANKED_PASSAGES
    params_model = ClaimCheckParams
    min_inputs = 1
    max_inputs = None

    def __init__(self, passages: PassageSet) -> None:
        self.passages = passages
        self._by_key: dict[tuple[str, str, int, int], Passage] = {
            (p.source_id, p.source_version, p.span.start, p.span.end): p for p in passages.passages
        }

    @property
    def bound_versions(self) -> BoundVersions:
        # Bound to the passage set it reads text from (snapshot + chunker).
        return BoundVersions(
            source_snapshot=self.passages.source_snapshot,
            index_version=self.passages.version,
            model_version=None,
        )

    def access_dependencies(
        self, params: ClaimCheckParams, access_scope: str, revoked: frozenset[str]
    ) -> frozenset[str]:
        return frozenset()  # reads only its inputs, which carry their own revocations

    def _text(self, e: Evidence) -> str | None:
        p = self._by_key.get((e.source_id, e.source_version, e.span.start, e.span.end))
        return p.text if p is not None and p.content_hash == e.content_hash else None

    async def run(
        self, params: ClaimCheckParams, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]:
        covering: list[Evidence] = []
        seen: set[tuple[str, int, int]] = set()
        for art in inputs:
            for e in art.evidence:
                key = (e.source_id, e.span.start, e.span.end)
                text = self._text(e)
                if key in seen or text is None:
                    continue
                claims = extract_claims(text, (params.attribute,))
                if claims:
                    seen.add(key)
                    covering.append(
                        e.model_copy(
                            update={
                                "extracted_claims": claims,
                                "coverage_type": CoverageType.CLAIM,
                                "retrieval_node_id": ctx.signature,
                                "created_at": ctx.now,
                            }
                        )
                    )
        return covering
