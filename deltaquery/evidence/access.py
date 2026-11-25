"""Access policy: who may read which scope, and which sources are revoked within a scope.

Authorization lives here, outside any model. The policy is versioned per scope; that version
is part of every node signature, so any revocation (or restoration) in a scope makes all prior
artifacts in that scope non-reusable. This is deliberately coarse: it can only cause extra
recomputation, never a stale reuse.

In-memory reference implementation; a real deployment would back this with its authz system.
"""

from __future__ import annotations

from collections import defaultdict

from deltaquery.canonical import hash_canonical


class AccessDenied(PermissionError):
    pass


class AccessPolicy:
    def __init__(self) -> None:
        self._grants: defaultdict[str, set[str]] = defaultdict(set)
        self._revoked: defaultdict[str, set[str]] = defaultdict(set)

    # -- administration -------------------------------------------------------------------

    def grant(self, principal: str, scope: str) -> None:
        self._grants[principal].add(scope)

    def revoke_scope(self, principal: str, scope: str) -> None:
        self._grants[principal].discard(scope)

    def revoke_source(self, scope: str, source_id: str) -> None:
        self._revoked[scope].add(source_id)

    def restore_source(self, scope: str, source_id: str) -> None:
        self._revoked[scope].discard(source_id)

    # -- checks ---------------------------------------------------------------------------

    def can_read(self, principal: str, scope: str) -> bool:
        return scope in self._grants.get(principal, set())

    def check(self, principal: str, scope: str) -> None:
        if not self.can_read(principal, scope):
            raise AccessDenied(f"principal {principal!r} may not read scope {scope!r}")

    def revoked_sources(self, scope: str) -> frozenset[str]:
        return frozenset(self._revoked.get(scope, set()))

    def version(self, scope: str) -> str:
        return hash_canonical(
            "access", {"scope": scope, "revoked": sorted(self._revoked.get(scope, set()))}
        )
