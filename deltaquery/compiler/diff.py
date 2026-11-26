"""Typed revision ops, computed deterministically by diffing two intents (never by the LLM)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from deltaquery.compiler.intent import Constraint, Intent


class RevisionKind(StrEnum):
    NEW = "new"  # no prior intent
    UNCHANGED = "unchanged"
    REPLACE_ENTITY = "replace_entity"
    ADD_ENTITY = "add_entity"
    REMOVE_ENTITY = "remove_entity"
    ADD_DIMENSION = "add_dimension"
    REMOVE_DIMENSION = "remove_dimension"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    CHANGE_TASK = "change_task"


class RevisionOp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: RevisionKind
    old: str | None = None
    new: str | None = None


def diff_intents(prior: Intent | None, new: Intent) -> tuple[RevisionOp, ...]:
    if prior is None:
        return (RevisionOp(kind=RevisionKind.NEW),)
    ops: list[RevisionOp] = []
    if prior.task != new.task:
        ops.append(RevisionOp(kind=RevisionKind.CHANGE_TASK, old=prior.task, new=new.task))

    removed = sorted(set(prior.entities) - set(new.entities))
    added = sorted(set(new.entities) - set(prior.entities))
    # Pair one-for-one swaps as replacements; leftovers are adds/removes.
    for old, new_e in zip(removed, added, strict=False):
        ops.append(RevisionOp(kind=RevisionKind.REPLACE_ENTITY, old=old, new=new_e))
    for old in removed[len(added) :]:
        ops.append(RevisionOp(kind=RevisionKind.REMOVE_ENTITY, old=old))
    for new_e in added[len(removed) :]:
        ops.append(RevisionOp(kind=RevisionKind.ADD_ENTITY, new=new_e))

    for d in sorted(set(new.dimensions) - set(prior.dimensions)):
        ops.append(RevisionOp(kind=RevisionKind.ADD_DIMENSION, new=d))
    for d in sorted(set(prior.dimensions) - set(new.dimensions)):
        ops.append(RevisionOp(kind=RevisionKind.REMOVE_DIMENSION, old=d))

    def fmt(c: Constraint) -> str:
        return f"{c.attribute} {c.op.value} {c.value_seconds}s"

    for c in sorted(set(new.constraints) - set(prior.constraints), key=lambda c: c.sort_key()):
        ops.append(RevisionOp(kind=RevisionKind.ADD_CONSTRAINT, new=fmt(c)))
    for c in sorted(set(prior.constraints) - set(new.constraints), key=lambda c: c.sort_key()):
        ops.append(RevisionOp(kind=RevisionKind.REMOVE_CONSTRAINT, old=fmt(c)))

    return tuple(ops) or (RevisionOp(kind=RevisionKind.UNCHANGED),)
