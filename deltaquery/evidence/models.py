"""Evidence and artifact (result manifest) models.

Identity rules:
- An `Evidence` row is identified by `evidence_id`, a hash over every field except `created_at`.
  `created_at` is bookkeeping, not content, so identical retrievals made at different times
  are the same evidence.
- An `Artifact` is the immutable result manifest of one executed node: the ordered evidence it
  produced plus the versions it was computed under. `artifact_hash` covers the ordered evidence
  ids and the output contract; downstream signatures depend on it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deltaquery.canonical import hash_canonical

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Span(BaseModel):
    model_config = _FROZEN

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class CoverageType(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"
    FUSED = "fused"


class Evidence(BaseModel):
    model_config = _FROZEN

    source_id: str
    source_version: str
    span: Span
    content_hash: str
    retrieval_node_id: str  # signature of the retrieval node that first produced it
    score: float
    extracted_claims: tuple[str, ...] = ()
    coverage_type: CoverageType
    created_at: datetime

    @field_validator("score")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("score must be finite")
        return v

    def identity(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"created_at"})

    @property
    def evidence_id(self) -> str:
        return hash_canonical("evidence", self.identity())


def compute_artifact_hash(output_contract: str, evidence: Sequence[Evidence]) -> str:
    return hash_canonical(
        "artifact",
        {"output_contract": output_contract, "evidence": [e.evidence_id for e in evidence]},
    )


class Artifact(BaseModel):
    model_config = _FROZEN

    signature: str
    operator: str
    operator_version: str
    output_contract: str
    access_scope: str
    source_snapshot: str | None
    index_version: str | None
    model_version: str | None
    input_signatures: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    artifact_hash: str
    created_at: datetime

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        expected = compute_artifact_hash(self.output_contract, self.evidence)
        if self.artifact_hash != expected:
            raise ValueError("artifact_hash does not match evidence (corrupt or tampered)")
        return self

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(e.evidence_id for e in self.evidence)
