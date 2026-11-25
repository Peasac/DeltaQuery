"""Unit tests: canonicalization, signatures, store immutability, plan validation, scope."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest
from conftest import CORPUS_PATH, SCOPE, Rig, make_rig, merge, plan_compare, retrieve

from deltaquery.canonical import (
    CanonicalizationError,
    canonical_json,
    hash_canonical,
    normalize_entity_id,
)
from deltaquery.evidence.corpus import Corpus, PassageSet, SourceDocument
from deltaquery.evidence.models import Artifact, compute_artifact_hash
from deltaquery.evidence.store import DeterminismViolation
from deltaquery.operators.base import ParameterError
from deltaquery.operators.bm25 import BM25Index, BM25Operator
from deltaquery.runtime.models import Plan, PlanValidationError
from deltaquery.runtime.signature import node_signature


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# -- canonicalization ---------------------------------------------------------------------


def test_canonical_json_is_order_and_zero_insensitive() -> None:
    assert canonical_json({"b": 1, "a": [0.0, 1.5]}) == canonical_json({"a": [-0.0, 1.5], "b": 1})
    assert canonical_json({"x": 1}) != canonical_json({"x": 1.0})  # int and float differ
    assert canonical_json({"x": True}) != canonical_json({"x": 1})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "a"}, {"$float": "x"}, {1, 2}])
def test_canonicalization_rejects_unhashable_values(bad: Any) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json(bad)


def test_hashes_are_domain_separated() -> None:
    assert hash_canonical("node", {"a": 1}) != hash_canonical("artifact", {"a": 1})


def test_entity_normalization() -> None:
    assert normalize_entity_id("  Google   Cloud ") == "google-cloud"
    with pytest.raises(ValueError):
        normalize_entity_id("   ")


# -- signatures ---------------------------------------------------------------------------


def test_signature_depends_on_every_versioned_field(index: BM25Index) -> None:
    base = retrieve("A", index, "aws")
    sig = node_signature(base, "1", [])
    assert node_signature(base.model_copy(update={"id": "Z"}), "1", []) == sig  # id excluded
    variants = [
        node_signature(base, "2", []),
        node_signature(base, "1", ["deadbeef"]),
        node_signature(base.model_copy(update={"access_scope": "other"}), "1", []),
        node_signature(base.model_copy(update={"source_snapshot": "s2"}), "1", []),
        node_signature(base.model_copy(update={"index_version": "i2"}), "1", []),
        node_signature(base.model_copy(update={"model_version": "m"}), "1", []),
        node_signature(retrieve("A", index, "aws", top_k=4), "1", []),
    ]
    assert sig not in variants and len(set(variants)) == len(variants)


def test_non_canonical_params_are_rejected(index: BM25Index) -> None:
    op = BM25Operator(index)
    with pytest.raises(ParameterError):
        op.parse_params({"query": "RPO aws", "top_k": 3, "entity": "AWS"})
    rig = make_rig(index)
    node = retrieve("A", index, "aws").model_copy(
        update={"canonical_parameters": {"query": "RPO", "top_k": 3, "entity": "aws"}}
    )
    with pytest.raises(PlanValidationError):
        rig.executor.check_plan(Plan(nodes=(node,), output_id="A"))


# -- plan validation ----------------------------------------------------------------------


def test_plan_rejects_cycles_unknown_inputs_and_scope_mixing(index: BM25Index) -> None:
    with pytest.raises(ValueError, match="cycle"):
        Plan(nodes=(merge("X", "Y"), merge("Y", "X")), output_id="X")
    with pytest.raises(ValueError, match="unknown"):
        Plan(nodes=(merge("X", "nope"),), output_id="X")
    with pytest.raises(ValueError, match="access scope"):
        Plan(
            nodes=(retrieve("A", index, "aws", scope="restricted"), merge("D", "A")),
            output_id="D",
        )


def test_layers_group_independent_nodes(index: BM25Index) -> None:
    layers = plan_compare(index, "aws", "azure").layers()
    assert [[n.id for n in layer] for layer in layers] == [["A", "B"], ["D"]]


def test_node_declaring_stale_index_is_rejected(index: BM25Index) -> None:
    changed = Corpus(
        [
            *list(Corpus.from_json(CORPUS_PATH).documents),
            SourceDocument(
                source_id="aws-new", entity="aws", title="t", text="new doc", access_scope=SCOPE
            ),
        ]
    )
    new_index = BM25Index(PassageSet(changed))
    assert new_index.snapshot_id != index.snapshot_id and new_index.version != index.version
    rig = make_rig(new_index)  # operator bound to the new snapshot
    stale = plan_compare(index, "aws", "azure")  # plan written against the old snapshot
    with pytest.raises(PlanValidationError, match="versions"):
        run(rig.executor.execute(stale))


# -- access scope -------------------------------------------------------------------------


def test_bm25_never_returns_other_scope(rig: Rig, index: BM25Index) -> None:
    plan = Plan(nodes=(retrieve("A", index, "aws", top_k=20),), output_id="A")
    r = run(rig.executor.execute(plan))
    assert "aws-internal-runbook" not in {e.source_id for e in r.output.evidence}
    restricted = Plan(nodes=(retrieve("A", index, "aws", scope="restricted"),), output_id="A")
    r2 = run(rig.executor.execute(restricted))
    assert r2.executed == {"A"}  # different scope never reuses
    assert {e.source_id for e in r2.output.evidence} == {"aws-internal-runbook"}


# -- store --------------------------------------------------------------------------------


def _first_artifact(rig: Rig, index: BM25Index) -> Artifact:
    plan = Plan(nodes=(retrieve("A", index, "aws"),), output_id="A")
    return run(rig.executor.execute(plan)).output  # type: ignore[no-any-return]


def test_store_roundtrip_preserves_content(rig: Rig, index: BM25Index) -> None:
    art = _first_artifact(rig, index)
    loaded = run(rig.store.get_artifact(art.signature))
    assert loaded == art
    assert [e.score for e in loaded.evidence] == [e.score for e in art.evidence]  # exact floats


def test_store_rows_are_immutable(rig: Rig, index: BM25Index) -> None:
    art = _first_artifact(rig, index)
    conn = rig.store._raw_connection_for_tests()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE evidence SET score = '0.0'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE artifacts SET artifact_hash = 'x' WHERE signature = ?", (art.signature,)
        )


def test_store_rejects_same_signature_with_different_content(rig: Rig, index: BM25Index) -> None:
    art = _first_artifact(rig, index)
    forged = Artifact.model_validate(
        {
            **art.model_dump(),
            "evidence": art.evidence[:1],
            "artifact_hash": compute_artifact_hash(art.output_contract, art.evidence[:1]),
        }
    )
    with pytest.raises(DeterminismViolation):
        run(rig.store.put_artifact(forged))


def test_tampered_manifest_fails_hash_check(rig: Rig, index: BM25Index) -> None:
    art = _first_artifact(rig, index)
    with pytest.raises(ValueError, match="artifact_hash"):
        Artifact.model_validate({**art.model_dump(), "evidence": art.evidence[:1]})
