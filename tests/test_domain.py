"""Domain configuration: a non-cloud domain drives validation, the LLM schema and prompt,
grounding, planning and claim extraction; the cloud default is unchanged."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import SCOPE
from pydantic import ValidationError

from deltaquery.compiler.compiler import IntentCompiler, system_prompt
from deltaquery.compiler.grounding import mentions
from deltaquery.compiler.intent import Intent, intent_json_schema
from deltaquery.compiler.planner import Planner, PlannerConfig
from deltaquery.domain import CLOUD, AttributeSpec, Domain, TopicSpec
from deltaquery.evidence.corpus import PassageSet
from deltaquery.operators.base import ParameterError
from deltaquery.operators.bm25 import BM25Index
from deltaquery.operators.claims import compile_patterns, extract_claims
from deltaquery.operators.coverage import ClaimCheckOperator
from deltaquery.operators.dense import DenseIndex

DB = Domain(
    name="databases",
    version="1",
    entity_noun="database engines",
    entity_short="engines",
    topics=(
        TopicSpec("replication", "replication replicas follower lag", (r"\breplicat\w*",)),
        TopicSpec("licensing", "license licensing terms", (r"\blicen[cs]\w*",), ttl_s=3600),
    ),
    attributes=(
        AttributeSpec(
            "failover_time",
            "automatic failover time promotion",
            (r"\bfailover\b",),
            r"(?:failover time|promotion time)",
        ),
    ),
)
ENGINES = {"pgx", "mydb"}
CTX: dict[str, Any] = {"entities": ENGINES, "domain": DB}


def db_intent(**kw: Any) -> Intent:
    raw = {"task": "describe", "entities": ["pgx"], "dimensions": ["replication"]} | kw
    return Intent.model_validate(raw, context=CTX)


def test_intent_vocabulary_comes_from_the_domain() -> None:
    i = db_intent(constraints=[{"attribute": "failover_time", "op": "le", "value_seconds": 30}])
    assert i.dimensions == ("replication",)
    with pytest.raises(ValidationError):
        db_intent(dimensions=["pricing"])  # a cloud topic is not a database topic
    with pytest.raises(ValidationError):
        db_intent(constraints=[{"attribute": "rpo", "op": "eq", "value_seconds": 0}])
    with pytest.raises(ValidationError):  # and the default (cloud) domain rejects DB topics
        Intent.model_validate({"task": "describe", "entities": ["a"], "dimensions": ["licensing"]})


def test_llm_schema_and_prompt_use_the_domain() -> None:
    schema = intent_json_schema(ENGINES, DB)
    assert schema["$defs"]["Dimension"]["enum"] == ["replication", "licensing"]
    assert schema["$defs"]["Attribute"]["enum"] == ["failover_time"]
    prompt = system_prompt(DB)
    assert "database engines" in prompt and "replication, licensing" in prompt
    assert "disaster_recovery" not in prompt and "RPO" not in prompt


def test_grounding_uses_the_domain_keywords() -> None:
    m = mentions("What about licensing and failover for MyDB?", ENGINES, DB)
    assert m.entities == {"mydb"}
    assert m.dimensions == {"licensing"} and m.attributes == {"failover_time"}
    assert mentions("pricing and DR", ENGINES, DB).empty  # cloud words mean nothing here


def test_compiler_validates_against_the_domain() -> None:
    class Scripted:
        model_version = "scripted@0"

        async def complete_json(self, **_: Any) -> dict[str, Any]:
            return {"task": "describe", "entities": ["pgx"], "dimensions": ["licensing"]}

    c = IntentCompiler(Scripted(), ENGINES, domain=DB)
    r = asyncio.run(c.compile("Describe pgx licensing"))
    assert r.intent.dimensions == ("licensing",)


def test_planner_uses_domain_queries_and_ttls(
    index: BM25Index, dense_index: DenseIndex, passages: PassageSet
) -> None:
    p = Planner(PlannerConfig(bm25=index, dense=dense_index, access_scope=SCOPE, domain=DB))
    i = db_intent(
        dimensions=["replication", "licensing"],
        constraints=[{"attribute": "failover_time", "op": "le", "value_seconds": 30}],
    )
    nodes = {n.id: n for n in p.build(i).nodes}

    def q(node_id: str) -> list[str]:
        return sorted(nodes[node_id].canonical_parameters["query"].split())

    assert q("pgx__replication__lex") == sorted(DB.topic("replication").query.split())
    assert nodes["pgx__licensing__lex"].max_age_s == 3600
    assert nodes["pgx__replication__lex"].max_age_s is None
    assert nodes["pgx__failover_time__check"].canonical_parameters == {"attribute": "failover_time"}
    assert q("pgx__failover_time__lex") == sorted(DB.attribute("failover_time").query.split())


def test_claim_labels_come_from_the_domain_and_version_the_operator(
    passages: PassageSet,
) -> None:
    cloud = ClaimCheckOperator(passages)
    db = ClaimCheckOperator(passages, DB)
    assert cloud.version == ClaimCheckOperator.version  # default domain: historical version
    assert db.version != cloud.version  # different labels -> different signatures
    assert db.parse_params({"attribute": "failover_time"}).attribute == "failover_time"
    with pytest.raises(ParameterError):
        db.parse_params({"attribute": "rpo"})
    with pytest.raises(ParameterError):
        cloud.parse_params({"attribute": "failover_time"})
    pats = compile_patterns(DB.claim_labels)
    assert extract_claims("Failover time is 30 seconds.", ("failover_time",), pats) == (
        "failover_time=30s",
    )
    assert extract_claims("Failover time is about 30 seconds.", ("failover_time",), pats) == ()


def test_domain_fingerprint_tracks_every_field() -> None:
    assert CLOUD.fingerprint == CLOUD.fingerprint
    changed = Domain(
        name=DB.name,
        version=DB.version,
        entity_noun=DB.entity_noun,
        entity_short=DB.entity_short,
        topics=DB.topics,
        attributes=(AttributeSpec("failover_time", "x", (r"\bfailover\b",), r"(?:failover time)"),),
    )
    assert changed.fingerprint != DB.fingerprint
    with pytest.raises(ValueError):
        Domain("d", "1", "xs", "xs", (TopicSpec("a", "q", ()), TopicSpec("a", "q", ())), ())
