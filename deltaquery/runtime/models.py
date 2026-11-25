"""Plan and node models. Dependencies are declared here, never inferred from model output."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class FreshnessPolicy(StrEnum):
    # Valid exactly as long as the declared source snapshot / index version are current.
    SNAPSHOT = "snapshot"
    # Additionally expires `max_age_s` after the content was last computed or re-verified.
    TTL = "ttl"


class Node(BaseModel):
    model_config = _FROZEN

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")  # plan-local; not part of signature
    operator: str
    canonical_parameters: dict[str, JsonValue]
    input_ids: tuple[str, ...] = ()
    source_snapshot: str | None
    index_version: str | None
    model_version: str | None
    access_scope: str = Field(min_length=1)
    freshness_policy: FreshnessPolicy = FreshnessPolicy.SNAPSHOT
    max_age_s: int | None = Field(default=None, gt=0)
    output_contract: str
    # Gate: if the named node's output is non-empty, this node is skipped (empty output).
    # Used for evidence sufficiency: retrieval runs only when the claim check found nothing.
    skip_if_nonempty: str | None = None

    @model_validator(mode="after")
    def _freshness(self) -> Self:
        if (self.freshness_policy is FreshnessPolicy.TTL) != (self.max_age_s is not None):
            raise ValueError("max_age_s is required for, and only for, the ttl policy")
        return self


def _deps(n: Node) -> tuple[str, ...]:
    """Ordering dependencies: data inputs plus the gate, if any."""
    return n.input_ids + ((n.skip_if_nonempty,) if n.skip_if_nonempty else ())


class PlanValidationError(ValueError):
    pass


class Plan(BaseModel):
    model_config = _FROZEN

    nodes: tuple[Node, ...]
    output_id: str

    @model_validator(mode="after")
    def _validate(self) -> Self:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("duplicate node id")
        by_id = {n.id: n for n in self.nodes}
        if self.output_id not in by_id:
            raise PlanValidationError(f"output node {self.output_id!r} not in plan")
        for n in self.nodes:
            for dep in _deps(n):
                if dep not in by_id:
                    raise PlanValidationError(f"node {n.id!r} depends on unknown {dep!r}")
                if by_id[dep].access_scope != n.access_scope:
                    # Fail closed: no mixing evidence across access scopes inside one plan.
                    raise PlanValidationError(f"node {n.id!r} crosses access scope via {dep!r}")
        self.layers()  # raises on cycles
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def layers(self) -> list[list[Node]]:
        """Topological layers (Kahn). Nodes within a layer are independent; sorted by id."""
        remaining = {n.id: set(_deps(n)) for n in self.nodes}
        by_id = {n.id: n for n in self.nodes}
        done: set[str] = set()
        out: list[list[Node]] = []
        while remaining:
            ready = sorted(nid for nid, deps in remaining.items() if deps <= done)
            if not ready:
                raise PlanValidationError("plan contains a cycle")
            out.append([by_id[nid] for nid in ready])
            done.update(ready)
            for nid in ready:
                del remaining[nid]
        return out
