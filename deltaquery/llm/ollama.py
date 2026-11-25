"""Ollama client over plain HTTP (stdlib only), using schema-constrained structured output."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from deltaquery.llm.base import LLMError

DEFAULT_URL = "http://localhost:11434"


def _post(url: str, payload: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode())
            return body
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(f"ollama request to {url} failed: {exc}") from exc


class OllamaLLM:
    """Use `await OllamaLLM.connect(model)`; it resolves and optionally pins the model digest."""

    def __init__(self, model: str, digest: str, base_url: str = DEFAULT_URL) -> None:
        self.model = model
        self.digest = digest
        self.base_url = base_url.rstrip("/")

    @property
    def model_version(self) -> str:
        return f"ollama/{self.model}@{self.digest}"

    @classmethod
    async def connect(
        cls,
        model: str,
        *,
        base_url: str = DEFAULT_URL,
        expected_digest: str | None = None,
        timeout_s: float = 10.0,
    ) -> OllamaLLM:
        tags = await asyncio.to_thread(_post, f"{base_url}/api/tags", None, timeout_s)
        for m in tags.get("models", []):
            if m.get("name") == model:
                digest = str(m["digest"])
                if expected_digest is not None and digest != expected_digest:
                    raise LLMError(f"{model}: digest {digest} != pinned {expected_digest}")
                return cls(model, digest, base_url)
        raise LLMError(f"model {model!r} is not installed in ollama")

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 0},
        }
        resp = await asyncio.wait_for(
            asyncio.to_thread(_post, f"{self.base_url}/api/chat", payload, timeout_s),
            timeout=timeout_s + 1,
        )
        try:
            out = json.loads(resp["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"model output is not JSON: {resp!r:.300}") from exc
        if not isinstance(out, dict):
            raise LLMError("model output is not a JSON object")
        return out
