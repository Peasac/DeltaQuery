"""Typed intent: the only thing the LLM produces. Closed vocabularies everywhere.

Canonical form: entities, dimensions and constraints are deduplicated and sorted, so
"AWS vs Azure" and "Azure vs AWS" compile to the same intent and the same plan.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from deltaquery.canonical import normalize_entity_id

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Task(StrEnum):
    COMPARE = "compare"
    DESCRIBE = "describe"


class Dimension(StrEnum):
    DISASTER_RECOVERY = "disaster_recovery"
    PRICING = "pricing"
    BACKUP = "backup"
    REGIONS = "regions"
    ENCRYPTION = "encryption"


class Attribute(StrEnum):
    RPO = "rpo"
    RTO = "rto"


class Comparison(StrEnum):
    EQ = "eq"
    LE = "le"


class Constraint(BaseModel):
    model_config = _FROZEN

    attribute: Attribute
    op: Comparison
    value_seconds: int = Field(ge=0)

    def sort_key(self) -> tuple[str, str, int]:
        return (self.attribute.value, self.op.value, self.value_seconds)


class Intent(BaseModel):
    """Validate with context={"entities": <allowed set>} to enforce the entity vocabulary."""

    model_config = _FROZEN

    task: Task
    entities: tuple[str, ...] = Field(min_length=1, max_length=4)
    dimensions: tuple[Dimension, ...] = Field(min_length=1)
    constraints: tuple[Constraint, ...] = ()

    @field_validator("entities")
    @classmethod
    def _entities(cls, v: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        norm = tuple(sorted({normalize_entity_id(e) for e in v}))
        allowed = (info.context or {}).get("entities")
        if allowed is not None:
            unknown = set(norm) - set(allowed)
            if unknown:
                raise ValueError(f"unknown entities: {sorted(unknown)}")
        return norm

    @field_validator("dimensions")
    @classmethod
    def _dims(cls, v: tuple[Dimension, ...]) -> tuple[Dimension, ...]:
        return tuple(sorted(set(v), key=lambda d: d.value))

    @field_validator("constraints")
    @classmethod
    def _constraints(cls, v: tuple[Constraint, ...]) -> tuple[Constraint, ...]:
        return tuple(sorted(set(v), key=Constraint.sort_key))

    @model_validator(mode="after")
    def _task_arity(self) -> Self:
        if self.task is Task.COMPARE and len(self.entities) < 2:
            raise ValueError("compare needs at least two entities")
        return self


def intent_json_schema(entities: Iterable[str]) -> dict[str, Any]:
    """JSON schema handed to the LLM for constrained decoding (entity enum injected)."""
    schema = Intent.model_json_schema()
    schema["properties"]["entities"]["items"] = {"enum": sorted(entities)}
    return schema


def render_question(intent: Intent) -> str:
    """Self-contained question text built from the intent (no LLM). Revisions like
    "Replace Azure with GCP" are not questions on their own; this is."""
    names = ", ".join(intent.entities)
    topics = ", ".join(d.value.replace("_", " ") for d in intent.dimensions)
    verb = "Compare" if intent.task is Task.COMPARE else "Describe"
    q = f"{verb} {names} on: {topics}."
    if intent.constraints:
        cons = "; ".join(
            f"{c.attribute.value.upper()} {'=' if c.op is Comparison.EQ else '<='} "
            f"{c.value_seconds} seconds"
            for c in intent.constraints
        )
        q += f" Requirements: {cons}."
    return q
