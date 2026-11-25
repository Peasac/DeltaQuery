"""HTTP API (project.md §11) over one Runtime.

AUTHENTICATION IS A PLACEHOLDER: the caller's principal is read from the `X-Principal` header
and trusted. That is acceptable only for local development. Authorization itself is real and
fails closed (AccessPolicy), but whoever can set the header can claim any principal. Put a real
authenticating proxy or auth dependency in front of this before exposing it anywhere.

Sessions belong to the principal that created them; any other principal gets 404 (existence is
not revealed). Evidence is served only if the caller may read a scope whose artifacts reference
it and the source is not revoked in that scope.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from deltaquery.api.views import (
    EvidenceView,
    PlanView,
    TurnView,
    evidence_view,
    plan_view,
    turn_view,
)
from deltaquery.compiler.compiler import CompileError
from deltaquery.evidence.access import AccessDenied
from deltaquery.runtime.answer import SpanResolver
from deltaquery.runtime.models import PlanValidationError
from deltaquery.sdk import Runtime
from deltaquery.session import Session, SupersededError, VersionConflict


class CreateSession(BaseModel):
    scope: str = Field(min_length=1)


class SessionCreated(BaseModel):
    session_id: str
    scope: str
    version: int


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    expected_version: int | None = None


class TraceView(BaseModel):
    session_id: str
    scope: str
    version: int
    turns: list[TurnView]


@dataclass
class _Entry:
    session: Session
    principal: str
    scope: str


def principal_header(x_principal: Annotated[str | None, Header()] = None) -> str:
    if not x_principal:
        raise HTTPException(401, "missing X-Principal header")
    return x_principal


Principal = Annotated[str, Depends(principal_header)]


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="DeltaQuery", version="0.0.1")
    sessions: dict[str, _Entry] = {}
    resolver = SpanResolver(runtime.passages)

    def owned(session_id: str, principal: str) -> _Entry:
        entry = sessions.get(session_id)
        if entry is None or entry.principal != principal:
            raise HTTPException(404, "session not found")
        try:  # re-check on every call: grants can be revoked mid-session
            runtime.policy.check(principal, entry.scope)
        except AccessDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        return entry

    @app.post("/sessions", status_code=201)
    async def create_session(body: CreateSession, principal: Principal) -> SessionCreated:
        try:
            session = runtime.session(scope=body.scope, principal=principal)
        except AccessDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        sid = secrets.token_urlsafe(12)
        sessions[sid] = _Entry(session, principal, body.scope)
        return SessionCreated(session_id=sid, scope=body.scope, version=0)

    @app.post("/sessions/{session_id}/queries")
    async def query(session_id: str, body: QueryRequest, principal: Principal) -> TurnView:
        entry = owned(session_id, principal)
        try:
            turn = await entry.session.ask(body.text, expected_version=body.expected_version)
        except VersionConflict as exc:
            raise HTTPException(409, {"error": "version_conflict", "detail": str(exc)}) from exc
        except SupersededError as exc:
            raise HTTPException(409, {"error": "superseded", "detail": str(exc)}) from exc
        except AccessDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        except (CompileError, PlanValidationError) as exc:
            raise HTTPException(422, {"error": "cannot_plan", "detail": str(exc)}) from exc
        return turn_view(session_id, turn)

    @app.get("/sessions/{session_id}/plans/{version}")
    async def get_plan(session_id: str, version: int, principal: Principal) -> PlanView:
        entry = owned(session_id, principal)
        turns = entry.session.turns
        if not 1 <= version <= len(turns):
            raise HTTPException(404, "no such version")
        prev = turns[version - 2] if version >= 2 else None
        return plan_view(session_id, turns[version - 1], prev, resolver)

    @app.get("/sessions/{session_id}/trace")
    async def get_trace(session_id: str, principal: Principal) -> TraceView:
        entry = owned(session_id, principal)
        return TraceView(
            session_id=session_id,
            scope=entry.scope,
            version=entry.session.version,
            turns=[turn_view(session_id, t) for t in entry.session.turns],
        )

    @app.get("/evidence/{evidence_id}")
    async def get_evidence(evidence_id: str, principal: Principal) -> EvidenceView:
        evidence = await runtime.store.get_evidence(evidence_id)
        if evidence is not None:
            scopes = await runtime.store.scopes_for_evidence(evidence_id)
            for scope in sorted(scopes):
                if runtime.policy.can_read(principal, scope) and (
                    evidence.source_id not in runtime.policy.revoked_sources(scope)
                ):
                    return evidence_view(evidence, resolver)
        raise HTTPException(404, "evidence not found")  # same answer whether absent or denied

    return app
