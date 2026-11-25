"""The single LLM interface. Everything above this layer is vendor-neutral."""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    """Transport failure, timeout, or output that is not valid JSON."""


class StructuredLLM(Protocol):
    @property
    def model_version(self) -> str:
        """Identifies weights + quantization (e.g. a content digest), not just a name."""
        ...

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        """Return a JSON object constrained to `schema`. Caller still validates it."""
        ...
