"""Common operator contract: typed params in, Evidence list out, declared output contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from deltaquery.canonical import canonicalize
from deltaquery.evidence.models import Artifact, Evidence

P = TypeVar("P", bound=BaseModel)


@dataclass(frozen=True)
class ExecutionContext:
    signature: str  # signature of the node being executed
    access_scope: str
    now: datetime


@dataclass(frozen=True)
class BoundVersions:
    """The versions an operator instance is actually bound to. Nodes must declare the same."""

    source_snapshot: str | None
    index_version: str | None
    model_version: str | None


class ParameterError(ValueError):
    pass


class Operator(ABC, Generic[P]):
    name: ClassVar[str]
    version: ClassVar[str]  # bump whenever behaviour changes; part of every signature
    output_contract: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]
    min_inputs: ClassVar[int] = 0
    max_inputs: ClassVar[int | None] = 0

    @property
    @abstractmethod
    def bound_versions(self) -> BoundVersions: ...

    def parse_params(self, raw: dict[str, Any]) -> P:
        """Validate params and require that they are already in canonical form.

        Rejecting non-canonical input (instead of silently normalizing) keeps the signature
        computed from `raw` equal to the one computed from the parsed params.
        """
        params = self.params_model.model_validate(raw)
        if canonicalize(params.model_dump(mode="python")) != canonicalize(raw):
            raise ParameterError(f"{self.name}: parameters are not canonical: {raw!r}")
        return params  # type: ignore[return-value]

    @abstractmethod
    async def run(
        self, params: P, inputs: Sequence[Artifact], ctx: ExecutionContext
    ) -> Sequence[Evidence]: ...
