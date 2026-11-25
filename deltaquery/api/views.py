"""Pure functions turning session state into API/trace-UI payloads (Pydantic models)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, JsonValue

from deltaquery.evidence.models import Evidence
from deltaquery.runtime.answer import Answer, ProvenanceError, SpanResolver
from deltaquery.runtime.executor import NodeStatus
from deltaquery.session import Turn


class EvidenceView(BaseModel):
    evidence_id: str
    source_id: str
    source_version: str
    span: tuple[int, int]
    score: float
    coverage_type: str
    quote: str | None  # None if the source text could not be verified


class NodeView(BaseModel):
    id: str
    operator: str
    inputs: list[str]
    status: str  # "executed" | "reused"
    layer: int
    signature: str
    reuse_rejected: str | None
    freshness_policy: str
    max_age_s: int | None
    params: dict[str, JsonValue]
    evidence: list[EvidenceView]


class RemovedNodeView(BaseModel):
    id: str
    operator: str
    inputs: list[str]
    consumed_by: list[str]  # nodes of the previous plan that consumed it
    layer: int  # layer in the previous plan


class PlanView(BaseModel):
    session_id: str
    version: int
    output_id: str
    nodes: list[NodeView]
    removed: list[RemovedNodeView]


class CitationView(BaseModel):
    ref: int
    evidence_id: str
    source_id: str
    span: tuple[int, int]
    quote: str


class SentenceView(BaseModel):
    text: str
    citations: list[CitationView]


class AnswerView(BaseModel):
    abstained: bool
    reason: str | None
    sentences: list[SentenceView]
    model_version: str | None


class TurnView(BaseModel):
    session_id: str
    version: int
    trace_id: str
    text: str
    intent: dict[str, Any]
    ops: list[dict[str, Any]]
    answer: AnswerView | None
    trace: dict[str, Any]


def evidence_view(e: Evidence, resolver: SpanResolver) -> EvidenceView:
    try:
        quote: str | None = resolver.text_for(e)
    except ProvenanceError:
        quote = None
    return EvidenceView(
        evidence_id=e.evidence_id,
        source_id=e.source_id,
        source_version=e.source_version,
        span=(e.span.start, e.span.end),
        score=e.score,
        coverage_type=e.coverage_type.value,
        quote=quote,
    )


def answer_view(a: Answer | None) -> AnswerView | None:
    if a is None:
        return None
    return AnswerView(
        abstained=a.abstained,
        reason=a.reason,
        model_version=a.model_version,
        sentences=[
            SentenceView(
                text=s.text,
                citations=[
                    CitationView(
                        ref=c.ref,
                        evidence_id=c.evidence_id,
                        source_id=c.source_id,
                        span=(c.span.start, c.span.end),
                        quote=c.quote,
                    )
                    for c in s.citations
                ],
            )
            for s in a.sentences
        ],
    )


def turn_view(session_id: str, turn: Turn) -> TurnView:
    v = turn.trace.version
    return TurnView(
        session_id=session_id,
        version=v,
        trace_id=f"{session_id}:{v}",
        text=turn.text,
        intent=turn.intent.model_dump(mode="json"),
        ops=[o.model_dump(mode="json") for o in turn.compiled.ops],
        answer=answer_view(turn.answer),
        trace=turn.trace.to_dict(),
    )


def plan_view(
    session_id: str, turn: Turn, previous: Turn | None, resolver: SpanResolver
) -> PlanView:
    layer_of = {n.id: i for i, layer in enumerate(turn.plan.layers()) for n in layer}
    status = {t.node_id: t for t in turn.result.trace}
    nodes = [
        NodeView(
            id=n.id,
            operator=n.operator,
            inputs=list(n.input_ids),
            status=status[n.id].status.value,
            layer=layer_of[n.id],
            signature=status[n.id].signature,
            reuse_rejected=status[n.id].reuse_rejected,
            freshness_policy=n.freshness_policy.value,
            max_age_s=n.max_age_s,
            params=dict(n.canonical_parameters),
            evidence=[evidence_view(e, resolver) for e in turn.result.artifacts[n.id].evidence],
        )
        for n in turn.plan.nodes
    ]
    removed: list[RemovedNodeView] = []
    if previous is not None:
        prev_layer = {n.id: i for i, layer in enumerate(previous.plan.layers()) for n in layer}
        gone = set(turn.removed)
        for n in previous.plan.nodes:
            if n.id in gone:
                removed.append(
                    RemovedNodeView(
                        id=n.id,
                        operator=n.operator,
                        inputs=list(n.input_ids),
                        consumed_by=sorted(
                            m.id for m in previous.plan.nodes if n.id in m.input_ids
                        ),
                        layer=prev_layer[n.id],
                    )
                )
    return PlanView(
        session_id=session_id,
        version=turn.trace.version,
        output_id=turn.plan.output_id,
        nodes=nodes,
        removed=removed,
    )


__all__ = ["NodeStatus", "PlanView", "TurnView", "plan_view", "turn_view"]
