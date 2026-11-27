"""Ollama client over plain HTTP (stdlib asyncio streams), using schema-constrained output.

Requests run on the event loop, not in a thread, so cancelling the awaiting task (a
superseded session epoch) or hitting the timeout closes the socket; Ollama aborts generation
when its client disconnects, so a superseded question stops using the model.
Only plain http:// is supported (a local Ollama); do not point this at a remote host.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from urllib.parse import urlsplit

from deltaquery.llm.base import LLMError

DEFAULT_URL = "http://localhost:11434"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
CRLF = b"\r\n"


def _dechunk(body: bytes) -> bytes:
    out = bytearray()
    while True:
        line, sep, rest = body.partition(CRLF)
        if not sep:
            raise LLMError("truncated chunked response")
        try:
            size = int(line.split(b";")[0].strip(), 16)
        except ValueError as exc:
            raise LLMError("malformed chunked response") from exc
        if size == 0:
            return bytes(out)
        if len(rest) < size:
            raise LLMError("truncated chunked response")
        out += rest[:size]
        body = rest[size + len(CRLF) :]


async def _read_all(reader: asyncio.StreamReader) -> bytes:
    raw = bytearray()
    while True:
        more = await reader.read(65536)
        if not more:
            return bytes(raw)
        raw += more
        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMError("ollama response too large")


def _parse_response(raw: bytes, url: str) -> dict[str, Any]:
    header, sep, body = raw.partition(CRLF + CRLF)
    if not sep:
        raise LLMError(f"malformed HTTP response from {url}")
    lines = header.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        raise LLMError(f"malformed status line from {url}: {lines[0]!r:.100}")
    status = int(parts[1])
    headers = {k.strip().lower(): v.strip() for k, _, v in (h.partition(":") for h in lines[1:])}
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body = _dechunk(body)
    if status != 200:
        raise LLMError(f"ollama returned HTTP {status}: {body[:200]!r}")
    try:
        out = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMError(f"ollama returned non-JSON body: {body[:200]!r}") from exc
    if not isinstance(out, dict):
        raise LLMError("ollama returned a non-object body")
    return out


async def _exchange(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    u = urlsplit(url)
    if u.scheme != "http" or not u.hostname:
        raise LLMError(f"unsupported ollama url {url!r} (plain http only)")
    data = b"" if payload is None else json.dumps(payload).encode()
    method = "GET" if payload is None else "POST"
    head = (
        f"{method} {u.path or '/'} HTTP/1.1\r\n"
        f"Host: {u.hostname}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(data)}\r\n"
        "Connection: close\r\n\r\n"
    )
    try:
        reader, writer = await asyncio.open_connection(u.hostname, u.port or 80)
    except OSError as exc:
        raise LLMError(f"ollama connection to {url} failed: {exc}") from exc
    try:
        writer.write(head.encode() + data)
        await writer.drain()
        raw = await _read_all(reader)
    except OSError as exc:
        raise LLMError(f"ollama request to {url} failed: {exc}") from exc
    finally:  # also on cancellation/timeout: the disconnect is what stops generation
        writer.close()
        with contextlib.suppress(Exception):
            await asyncio.shield(writer.wait_closed())
    return _parse_response(raw, url)


async def _post(url: str, payload: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(_exchange(url, payload), timeout=timeout_s)
    except TimeoutError as exc:
        raise LLMError(f"ollama request to {url} timed out after {timeout_s}s") from exc


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
        tags = await _post(f"{base_url.rstrip('/')}/api/tags", None, timeout_s)
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
        resp = await _post(f"{self.base_url}/api/chat", payload, timeout_s)
        try:
            out = json.loads(resp["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"model output is not JSON: {resp!r:.300}") from exc
        if not isinstance(out, dict):
            raise LLMError("model output is not a JSON object")
        return out
