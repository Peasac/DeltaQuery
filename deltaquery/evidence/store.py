"""SQLite evidence store: immutable evidence rows, artifact manifests and dependency edges.

Content tables (evidence, artifacts, artifact_evidence, artifact_inputs) are insert-only: UPDATE
is blocked by triggers. Re-storing an artifact under an existing signature is allowed only if its
content hash is identical; anything else is a determinism violation and raises.

`artifact_usage` is the one mutable table. It holds bookkeeping only (last use, last
re-verification, hit count) for freshness and eviction, never content.

DELETE is allowed for eviction. Eviction is always safe under content addressing: an evicted
node is simply recomputed and, being deterministic, yields the same artifact hash, so its
dependents' signatures and stored artifacts remain valid.

The stdlib `sqlite3` driver is blocking, so every call runs in a worker thread behind a lock.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar

from deltaquery.evidence.models import Artifact, CoverageType, Evidence, Span

T = TypeVar("T")

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
    signature             TEXT PRIMARY KEY,
    operator              TEXT NOT NULL,
    operator_version      TEXT NOT NULL,
    output_contract       TEXT NOT NULL,
    access_scope          TEXT NOT NULL,
    access_policy_version TEXT NOT NULL,
    source_snapshot       TEXT,
    index_version         TEXT,
    model_version         TEXT,
    artifact_hash         TEXT NOT NULL,
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_scope ON artifacts(access_scope);
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
CREATE TABLE IF NOT EXISTS artifact_usage (
    signature        TEXT PRIMARY KEY REFERENCES artifacts(signature),
    last_used_at     TEXT NOT NULL,
    last_verified_at TEXT NOT NULL,
    hits             INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS evidence_immutable BEFORE UPDATE ON evidence
BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_immutable BEFORE UPDATE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact rows are immutable'); END;
"""

_EVIDENCE_COLS = (
    "e.source_id, e.source_version, e.span_start, e.span_end, e.content_hash,"
    " e.retrieval_node_id, e.score, e.extracted_claims, e.coverage_type, e.created_at"
)


class DeterminismViolation(RuntimeError):
    """Same signature, different content: the operator or its inputs are not deterministic."""


def _evidence_from_row(r: tuple[Any, ...]) -> Evidence:
    return Evidence(
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


class EvidenceStore:
    """`cache_size` > 0 keeps a read-through LRU of artifacts in process. That is safe because
    artifact rows are immutable: an entry can only go stale by eviction, which also purges it.
    Entries come from validated writes or validated loads. Usage bookkeeping (touches) is
    buffered and flushed in batches, before eviction, and on close."""

    def __init__(
        self, path: str | Path = ":memory:", *, cache_size: int = 4096, flush_every: int = 256
    ) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._cache_size = cache_size
        self._flush_every = flush_every
        self._mem: OrderedDict[str, Artifact] = OrderedDict()
        self._verified: dict[str, datetime] = {}
        self._pending: dict[str, tuple[datetime, int]] = {}  # signature -> (last use, hits)

    def _remember(self, a: Artifact) -> None:
        if self._cache_size <= 0:
            return
        self._mem[a.signature] = a
        self._mem.move_to_end(a.signature)
        while len(self._mem) > self._cache_size:
            self._mem.popitem(last=False)

    def clear_memory_cache(self) -> None:
        """Drop in-process state (as after a restart). Pending touches are kept."""
        self._mem.clear()
        self._verified.clear()

    async def flush(self) -> None:
        """Write buffered usage bookkeeping to SQLite."""
        if not self._pending:
            return
        batch, self._pending = self._pending, {}

        def fn(c: sqlite3.Connection) -> None:
            c.executemany(
                "UPDATE artifact_usage SET last_used_at = max(last_used_at, ?),"
                " hits = hits + ? WHERE signature = ?",
                [(t.isoformat(), n, sig) for sig, (t, n) in batch.items()],
            )

        await self._run(fn)

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
        await self.flush()
        await asyncio.to_thread(self._conn.close)

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def locked() -> T:
            with self._lock:
                return fn(self._conn)

        return await asyncio.to_thread(locked)

    # -- writes ---------------------------------------------------------------------------

    async def put_artifact(self, artifact: Artifact) -> None:
        """Insert, or (same signature, same content) record a re-verification."""
        inserted = await self._run(lambda c: self._put_artifact(c, artifact))
        when = artifact.created_at
        prev = self._verified.get(artifact.signature)
        self._verified[artifact.signature] = when if prev is None else max(prev, when)
        if inserted:
            self._remember(artifact)

    @staticmethod
    def _put_artifact(conn: sqlite3.Connection, a: Artifact) -> bool:
        cur = conn.cursor()
        now = a.created_at.isoformat()
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
                cur.execute(
                    "UPDATE artifact_usage SET last_verified_at = max(last_verified_at, ?),"
                    " last_used_at = max(last_used_at, ?) WHERE signature = ?",
                    (now, now, a.signature),
                )
                cur.execute("COMMIT")
                return False
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
                "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    a.signature,
                    a.operator,
                    a.operator_version,
                    a.output_contract,
                    a.access_scope,
                    a.access_policy_version,
                    a.source_snapshot,
                    a.index_version,
                    a.model_version,
                    a.artifact_hash,
                    now,
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
            cur.execute("INSERT INTO artifact_usage VALUES (?,?,?,0)", (a.signature, now, now))
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
        return True

    async def touch(self, signature: str, now: datetime) -> None:
        """Record a reuse (for LRU eviction). Buffered; see flush()."""
        prev = self._pending.get(signature)
        self._pending[signature] = (now, 1) if prev is None else (max(prev[0], now), prev[1] + 1)
        if signature in self._mem:
            self._mem.move_to_end(signature)
        if len(self._pending) >= self._flush_every:
            await self.flush()

    # -- reads ----------------------------------------------------------------------------

    async def get_artifact(self, signature: str) -> Artifact | None:
        """Load an artifact. The model re-verifies artifact_hash against stored evidence and
        raises ValueError if they disagree (corruption)."""
        hit = self._mem.get(signature)
        if hit is not None:
            self._mem.move_to_end(signature)
            return hit
        art = await self._run(lambda c: self._get_artifact(c, signature))
        if art is not None:
            self._remember(art)
        return art

    @staticmethod
    def _get_artifact(conn: sqlite3.Connection, signature: str) -> Artifact | None:
        row = conn.execute(
            "SELECT signature, operator, operator_version, output_contract, access_scope,"
            " access_policy_version, source_snapshot, index_version, model_version,"
            " artifact_hash, created_at FROM artifacts WHERE signature = ?",
            (signature,),
        ).fetchone()
        if row is None:
            return None
        ev_rows = conn.execute(
            f"SELECT {_EVIDENCE_COLS} FROM artifact_evidence ae"
            " JOIN evidence e ON e.evidence_id = ae.evidence_id"
            " WHERE ae.signature = ? ORDER BY ae.position",
            (signature,),
        ).fetchall()
        inputs = conn.execute(
            "SELECT input_signature FROM artifact_inputs WHERE signature = ? ORDER BY position",
            (signature,),
        ).fetchall()
        return Artifact(
            signature=row[0],
            operator=row[1],
            operator_version=row[2],
            output_contract=row[3],
            access_scope=row[4],
            access_policy_version=row[5],
            source_snapshot=row[6],
            index_version=row[7],
            model_version=row[8],
            input_signatures=tuple(r[0] for r in inputs),
            evidence=tuple(_evidence_from_row(r) for r in ev_rows),
            artifact_hash=row[9],
            created_at=datetime.fromisoformat(row[10]),
        )

    async def last_verified_at(self, signature: str) -> datetime | None:
        if signature in self._verified:
            return self._verified[signature]

        def fn(c: sqlite3.Connection) -> datetime | None:
            row = c.execute(
                "SELECT last_verified_at FROM artifact_usage WHERE signature = ?", (signature,)
            ).fetchone()
            return None if row is None else datetime.fromisoformat(row[0])

        return await self._run(fn)

    async def get_evidence(self, evidence_id: str) -> Evidence | None:
        def fn(c: sqlite3.Connection) -> Evidence | None:
            row = c.execute(
                f"SELECT {_EVIDENCE_COLS} FROM evidence e WHERE e.evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            return None if row is None else _evidence_from_row(row)

        return await self._run(fn)

    async def scopes_for_evidence(self, evidence_id: str) -> frozenset[str]:
        """Access scopes of the artifacts that reference this evidence (for authorization)."""

        def fn(c: sqlite3.Connection) -> frozenset[str]:
            rows = c.execute(
                "SELECT DISTINCT a.access_scope FROM artifact_evidence ae"
                " JOIN artifacts a ON a.signature = ae.signature WHERE ae.evidence_id = ?",
                (evidence_id,),
            ).fetchall()
            return frozenset(r[0] for r in rows)

        return await self._run(fn)

    async def lineage(self, signature: str) -> dict[str, Artifact | None]:
        """Provenance: the artifact and all its transitive inputs, inputs first.
        Evicted (or never stored) ancestors map to None."""
        out: dict[str, Artifact | None] = {}

        async def visit(sig: str) -> None:
            if sig in out:
                return
            art = await self.get_artifact(sig)
            if art is not None:
                for parent in art.input_signatures:
                    await visit(parent)
            out[sig] = art

        await visit(signature)
        return out

    async def count_artifacts(self, scope: str | None = None) -> int:
        def fn(c: sqlite3.Connection) -> int:
            if scope is None:
                n: int = c.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            else:
                n = c.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE access_scope = ?", (scope,)
                ).fetchone()[0]
            return n

        return await self._run(fn)

    async def count_evidence(self) -> int:
        return await self._run(
            lambda c: int(c.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        )

    # -- eviction -------------------------------------------------------------------------

    async def evict(self, *, scope: str, max_artifacts: int) -> int:
        """Keep at most `max_artifacts` in `scope` (least recently used go first), then drop
        evidence no artifact references. Returns the number of artifacts evicted."""
        if max_artifacts < 0:
            raise ValueError("max_artifacts must be >= 0")
        await self.flush()  # LRU order must include buffered uses
        evicted: list[str] = []

        def fn(c: sqlite3.Connection) -> int:
            cur = c.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                victims = [
                    r[0]
                    for r in cur.execute(
                        "SELECT a.signature FROM artifacts a"
                        " JOIN artifact_usage u ON u.signature = a.signature"
                        " WHERE a.access_scope = ?"
                        " ORDER BY u.last_used_at DESC, u.hits DESC, a.signature"
                        " LIMIT -1 OFFSET ?",
                        (scope, max_artifacts),
                    ).fetchall()
                ]
                for table in (
                    "artifact_evidence",
                    "artifact_inputs",
                    "artifact_usage",
                    "artifacts",
                ):
                    cur.executemany(
                        f"DELETE FROM {table} WHERE signature = ?", [(v,) for v in victims]
                    )
                cur.execute(
                    "DELETE FROM evidence WHERE evidence_id NOT IN"
                    " (SELECT evidence_id FROM artifact_evidence)"
                )
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            evicted.extend(victims)
            return len(victims)

        n = await self._run(fn)
        for sig in evicted:
            self._mem.pop(sig, None)
            self._verified.pop(sig, None)
        return n

    def _raw_connection_for_tests(self) -> sqlite3.Connection:
        return self._conn
