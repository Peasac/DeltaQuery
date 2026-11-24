"""SQLite evidence store: immutable evidence rows, artifact manifests and dependency edges.

Writes are insert-only. UPDATE on evidence/artifacts is blocked by triggers. Re-storing an
artifact under an existing signature is allowed only if its content hash is identical; anything
else is a determinism violation and raises (the signature promised the same result).

The stdlib `sqlite3` driver is blocking, so every call runs in a worker thread behind a lock.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from deltaquery.evidence.models import Artifact, CoverageType, Evidence, Span

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id       TEXT PRIMARY KEY,
    source_id         TEXT NOT NULL,
    source_version    TEXT NOT NULL,
    span_start        INTEGER NOT NULL,
    span_end          INTEGER NOT NULL,
    content_hash      TEXT NOT NULL,
    retrieval_node_id TEXT NOT NULL,
    score             TEXT NOT NULL,
    extracted_claims  TEXT NOT NULL,
    coverage_type     TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    signature        TEXT PRIMARY KEY,
    operator         TEXT NOT NULL,
    operator_version TEXT NOT NULL,
    output_contract  TEXT NOT NULL,
    access_scope     TEXT NOT NULL,
    source_snapshot  TEXT,
    index_version    TEXT,
    model_version    TEXT,
    artifact_hash    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_evidence (
    signature   TEXT NOT NULL REFERENCES artifacts(signature),
    position    INTEGER NOT NULL,
    evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
    PRIMARY KEY (signature, position)
);
CREATE TABLE IF NOT EXISTS artifact_inputs (
    signature       TEXT NOT NULL REFERENCES artifacts(signature),
    position        INTEGER NOT NULL,
    input_signature TEXT NOT NULL,
    PRIMARY KEY (signature, position)
);
CREATE TRIGGER IF NOT EXISTS evidence_immutable BEFORE UPDATE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_immutable BEFORE UPDATE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact rows are immutable'); END;
"""


class DeterminismViolation(RuntimeError):
    """Same signature, different content: the operator or its inputs are not deterministic."""


class EvidenceStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    # -- writes ---------------------------------------------------------------------------

    async def put_artifact(self, artifact: Artifact) -> None:
        await asyncio.to_thread(self._put_artifact, artifact)

    def _put_artifact(self, a: Artifact) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute(
                    "SELECT artifact_hash FROM artifacts WHERE signature = ?", (a.signature,)
                ).fetchone()
                if row is not None:
                    if row[0] != a.artifact_hash:
                        raise DeterminismViolation(
                            f"signature {a.signature[:12]} already stored with different content"
                        )
                    cur.execute("ROLLBACK")
                    return
                for e in a.evidence:
                    cur.execute(
                        "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            e.evidence_id,
                            e.source_id,
                            e.source_version,
                            e.span.start,
                            e.span.end,
                            e.content_hash,
                            e.retrieval_node_id,
                            repr(e.score),
                            json.dumps(list(e.extracted_claims)),
                            e.coverage_type.value,
                            e.created_at.isoformat(),
                        ),
                    )
                cur.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        a.signature,
                        a.operator,
                        a.operator_version,
                        a.output_contract,
                        a.access_scope,
                        a.source_snapshot,
                        a.index_version,
                        a.model_version,
                        a.artifact_hash,
                        a.created_at.isoformat(),
                    ),
                )
                cur.executemany(
                    "INSERT INTO artifact_evidence VALUES (?,?,?)",
                    [(a.signature, i, eid) for i, eid in enumerate(a.evidence_ids)],
                )
                cur.executemany(
                    "INSERT INTO artifact_inputs VALUES (?,?,?)",
                    [(a.signature, i, s) for i, s in enumerate(a.input_signatures)],
                )
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise

    # -- reads ----------------------------------------------------------------------------

    async def get_artifact(self, signature: str) -> Artifact | None:
        """Load an artifact. The model re-verifies artifact_hash against stored evidence."""
        return await asyncio.to_thread(self._get_artifact, signature)

    def _get_artifact(self, signature: str) -> Artifact | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT signature, operator, operator_version, output_contract, access_scope,"
                " source_snapshot, index_version, model_version, artifact_hash, created_at"
                " FROM artifacts WHERE signature = ?",
                (signature,),
            ).fetchone()
            if row is None:
                return None
            ev_rows = self._conn.execute(
                "SELECT e.source_id, e.source_version, e.span_start, e.span_end, e.content_hash,"
                " e.retrieval_node_id, e.score, e.extracted_claims, e.coverage_type, e.created_at"
                " FROM artifact_evidence ae JOIN evidence e ON e.evidence_id = ae.evidence_id"
                " WHERE ae.signature = ? ORDER BY ae.position",
                (signature,),
            ).fetchall()
            inputs = self._conn.execute(
                "SELECT input_signature FROM artifact_inputs WHERE signature = ? ORDER BY position",
                (signature,),
            ).fetchall()
        evidence = tuple(
            Evidence(
                source_id=r[0],
                source_version=r[1],
                span=Span(start=r[2], end=r[3]),
                content_hash=r[4],
                retrieval_node_id=r[5],
                score=float(r[6]),
                extracted_claims=tuple(json.loads(r[7])),
                coverage_type=CoverageType(r[8]),
                created_at=datetime.fromisoformat(r[9]),
            )
            for r in ev_rows
        )
        return Artifact(
            signature=row[0],
            operator=row[1],
            operator_version=row[2],
            output_contract=row[3],
            access_scope=row[4],
            source_snapshot=row[5],
            index_version=row[6],
            model_version=row[7],
            input_signatures=tuple(r[0] for r in inputs),
            evidence=evidence,
            artifact_hash=row[8],
            created_at=datetime.fromisoformat(row[9]),
        )

    async def count_artifacts(self) -> int:
        def _count() -> int:
            with self._lock:
                n: int = self._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            return n

        return await asyncio.to_thread(_count)

    def _raw_connection_for_tests(self) -> sqlite3.Connection:
        return self._conn
