"""HTTP API: §11 endpoints, ownership, fail-closed authorization, error mapping."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import CORPUS_PATH, SCOPE
from fastapi.testclient import TestClient

from deltaquery.api.app import create_app
from deltaquery.evidence.access import AccessPolicy
from deltaquery.evidence.corpus import Corpus
from deltaquery.llm.base import LLMError
from deltaquery.sdk import Runtime

ENTITIES = {"aws", "azure", "gcp"}
Q1 = {"task": "compare", "entities": ["aws", "azure"], "dimensions": ["disaster_recovery"]}
Q2 = {"task": "compare", "entities": ["aws", "gcp"], "dimensions": ["disaster_recovery"]}
ANSWER = {"sentences": [{"text": "See passage one.", "citations": [1]}], "abstain": False}
Q1_TEXT = "Compare AWS and Azure disaster recovery"
ALICE = {"X-Principal": "alice"}
BOB = {"X-Principal": "bob"}


class ScriptedLLM:
    model_version = "scripted/test@0"

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs

    async def complete_json(self, **_: Any) -> dict[str, Any]:
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out  # type: ignore[no-any-return]


@pytest.fixture
def api() -> Iterator[tuple[TestClient, Runtime, ScriptedLLM]]:
    llm = ScriptedLLM([Q1, ANSWER, Q2, ANSWER])
    policy = AccessPolicy()
    policy.grant("alice", SCOPE)
    policy.grant("bob", SCOPE)
    runtime = asyncio.run(
        Runtime.create(
            corpus=Corpus.from_json(CORPUS_PATH), llm=llm, policy=policy, entities=ENTITIES
        )
    )
    with TestClient(create_app(runtime)) as client:
        yield client, runtime, llm


def _session(client: TestClient, headers: dict[str, str] = ALICE) -> str:
    r = client.post("/sessions", json={"scope": SCOPE}, headers=headers)
    assert r.status_code == 201, r.text
    return str(r.json()["session_id"])


def test_full_flow_plan_diff_and_trace(api: tuple[TestClient, Runtime, ScriptedLLM]) -> None:
    client, _, _ = api
    sid = _session(client)
    r1 = client.post(
        f"/sessions/{sid}/queries", json={"text": "Compare AWS and Azure DR"}, headers=ALICE
    )
    assert r1.status_code == 200, r1.text
    r2 = client.post(
        f"/sessions/{sid}/queries",
        json={"text": "Replace Azure with GCP", "expected_version": 1},
        headers=ALICE,
    )
    body = r2.json()
    assert body["version"] == 2 and body["trace_id"] == f"{sid}:2"
    assert [o["kind"] for o in body["ops"]] == ["replace_entity"]
    assert "aws__evidence" in body["trace"]["reused_nodes"]
    cite = body["answer"]["sentences"][0]["citations"][0]
    assert cite["quote"]

    plan = client.get(f"/sessions/{sid}/plans/2", headers=ALICE).json()
    status = {n["id"]: n["status"] for n in plan["nodes"]}
    assert status["aws__disaster_recovery"] == "reused"
    assert status["gcp__disaster_recovery"] == "executed"
    removed = {n["id"]: n for n in plan["removed"]}
    assert removed["azure__evidence"]["consumed_by"] == ["answer_evidence"]
    assert all(e["quote"] for n in plan["nodes"] for e in n["evidence"])
    assert client.get(f"/sessions/{sid}/plans/1", headers=ALICE).json()["removed"] == []
    assert client.get(f"/sessions/{sid}/plans/3", headers=ALICE).status_code == 404

    trace = client.get(f"/sessions/{sid}/trace", headers=ALICE).json()
    assert trace["version"] == 2 and [t["version"] for t in trace["turns"]] == [1, 2]

    ev = client.get(f"/evidence/{cite['evidence_id']}", headers=ALICE)
    assert ev.status_code == 200 and ev.json()["quote"] == cite["quote"]


def test_auth_header_required_and_scope_enforced(
    api: tuple[TestClient, Runtime, ScriptedLLM],
) -> None:
    client, _, _ = api
    assert client.post("/sessions", json={"scope": SCOPE}).status_code == 401
    r = client.post("/sessions", json={"scope": "restricted"}, headers=ALICE)
    assert r.status_code == 403
    r = client.post("/sessions", json={"scope": SCOPE}, headers={"X-Principal": "mallory"})
    assert r.status_code == 403


def test_sessions_are_private_to_their_principal(
    api: tuple[TestClient, Runtime, ScriptedLLM],
) -> None:
    client, _, _ = api
    sid = _session(client)
    client.post(f"/sessions/{sid}/queries", json={"text": Q1_TEXT}, headers=ALICE)
    for method, path in [
        ("post", f"/sessions/{sid}/queries"),
        ("get", f"/sessions/{sid}/trace"),
        ("get", f"/sessions/{sid}/plans/1"),
    ]:
        kw: dict[str, Any] = {"headers": BOB}
        if method == "post":
            kw["json"] = {"text": Q1_TEXT}
        assert getattr(client, method)(path, **kw).status_code == 404


def test_revocation_mid_session_and_evidence_authz(
    api: tuple[TestClient, Runtime, ScriptedLLM],
) -> None:
    client, runtime, _ = api
    sid = _session(client)
    body = client.post(f"/sessions/{sid}/queries", json={"text": Q1_TEXT}, headers=ALICE).json()
    eid = body["answer"]["sentences"][0]["citations"][0]["evidence_id"]
    source = body["answer"]["sentences"][0]["citations"][0]["source_id"]

    # Another principal without the scope cannot read the evidence (404, not 403).
    assert client.get(f"/evidence/{eid}", headers={"X-Principal": "eve"}).status_code == 404
    # Revoking the source hides the evidence even from its original reader.
    runtime.policy.revoke_source(SCOPE, source)
    assert client.get(f"/evidence/{eid}", headers=ALICE).status_code == 404
    runtime.policy.restore_source(SCOPE, source)
    # Revoking the principal's scope blocks the session itself.
    runtime.policy.revoke_scope("alice", SCOPE)
    assert client.get(f"/sessions/{sid}/trace", headers=ALICE).status_code == 403


def test_error_mapping(api: tuple[TestClient, Runtime, ScriptedLLM]) -> None:
    client, _, llm = api
    sid = _session(client)
    r = client.post(
        f"/sessions/{sid}/queries", json={"text": Q1_TEXT, "expected_version": 5}, headers=ALICE
    )
    assert r.status_code == 409 and r.json()["detail"]["error"] == "version_conflict"
    llm.outputs[:0] = [LLMError("down"), LLMError("down")]
    r = client.post(f"/sessions/{sid}/queries", json={"text": Q1_TEXT}, headers=ALICE)
    assert r.status_code == 422 and r.json()["detail"]["error"] == "cannot_plan"
    assert client.get(f"/sessions/{sid}/trace", headers=ALICE).json()["version"] == 0
