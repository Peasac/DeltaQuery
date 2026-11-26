"""Typed intent: the only thing the LLM produces. Closed vocabularies everywhere.

Canonical form: entities, dimensions and constraints are deduplicated and sorted, so
"AWS vs Azure" and "Azure vs AWS" compile to the same intent and the same plan.

Topics (`dimensions`) and constraint attributes are plain strings checked against a `Domain`
(deltaquery/domain.py) passed in the validation context as {"domain": ...}; without one, the
default (cloud) domain applies. `Dimension` and `Attribute` remain as the cloud domain's names.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from deltaquery.canonical import normalize_entity_id
from deltaquery.domain import DEFAULT_DOMAIN, Domain

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Task(StrEnum):
    COMPARE = "compare"
    DESCRIBE = "describe"


def _domain(info: ValidationInfo) -> Domain:
    d = (info.context or {}).get("domain")
    return d if isinstance(d, Domain) else DEFAULT_DOMAIN


class Dimension(StrEnum):  # names in the cloud domain
    DISASTER_RECOVERY = "disaster_recovery"
    PRICING = "pricing"
    BACKUP = "backup"
    REGIONS = "regions"
    ENCRYPTION = "encryption"


class Attribute(StrEnum):  # names in the cloud domain
    RPO = "rpo"
    RTO = "rto"


class Comparison(StrEnum):
    EQ = "eq"
    LE = "le"


class Constraint(BaseModel):
    model_config = _FROZEN

    attribute: str
    op: Comparison
    value_seconds: int = Field(ge=0)

    @field_validator("attribute")
    @classmethod
    def _attribute(cls, v: str, info: ValidationInfo) -> str:
        v = str(v)
        if v not in _domain(info).attribute_names:
            raise ValueError(f"unknown attribute: {v!r}")
        return v

    def sort_key(self) -> tuple[str, str, int]:
        return (self.attribute, self.op.value, self.value_seconds)


class Intent(BaseModel):
    """Validate with context={"entities": <allowed set>} to enforce the entity vocabulary."""

    model_config = _FROZEN

    task: Task
    entities: tuple[str, ...] = Field(min_length=1, max_length=4)
    dimensions: tuple[str, ...] = Field(min_length=1)
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
    def _dims(cls, v: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        dims = {str(d) for d in v}
        unknown = dims - set(_domain(info).topic_names)
        if unknown:
            raise ValueError(f"unknown dimensions: {sorted(unknown)}")
        return tuple(sorted(dims))

    @field_validator("constraints")
    @classmethod
    def _constraints(cls, v: tuple[Constraint, ...]) -> tuple[Constraint, ...]:
        return tuple(sorted(set(v), key=Constraint.sort_key))

    @model_validator(mode="after")
    def _task_arity(self) -> Self:
        if self.task is Task.COMPARE and len(self.entities) < 2:
            raise ValueError("compare needs at least two entities")
        return self


def intent_json_schema(entities: Iterable[str], domain: Domain = DEFAULT_DOMAIN) -> dict[str, Any]:
    """JSON schema handed to the LLM for constrained decoding (entity, topic and attribute
    enums injected from the domain)."""
    schema = Intent.model_json_schema()
    schema["properties"]["entities"]["items"] = {"enum": sorted(entities)}
    defs = schema["$defs"]
    for title, values in (("Attribute", domain.attribute_names), ("Dimension", domain.topic_names)):
        defs[title] = {"enum": list(values), "title": title, "type": "string"}
    schema["properties"]["dimensions"]["items"] = {"$ref": "#/$defs/Dimension"}
    defs["Constraint"]["properties"]["attribute"] = {"$ref": "#/$defs/Attribute"}
    schema["$defs"] = dict(sorted(defs.items()))
    return schema


def render_question(intent: Intent) -> str:
    """Self-contained question text built from the intent (no LLM). Revisions like
    "Replace Azure with GCP" are not questions on their own; this is."""
    names = ", ".join(intent.entities)
    topics = ", ".join(d.replace("_", " ") for d in intent.dimensions)
    verb = "Compare" if intent.task is Task.COMPARE else "Describe"
    q = f"{verb} {names} on: {topics}."
    if intent.constraints:
        cons = "; ".join(
            f"{c.attribute.upper()} {'=' if c.op is Comparison.EQ else '<='} "
            f"{c.value_seconds} seconds"
            for c in intent.constraints
        )
        q += f" Requirements: {cons}."
    return q
