"""Ollama client transport, against an in-process fake server on loopback (no network).

The property that matters: cancelling the awaiting task (a superseded epoch) or a timeout
closes the connection, which is what makes Ollama stop generating."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from deltaquery.llm.base import LLMError
from deltaquery.llm.ollama import OllamaLLM

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _read_request(reader: asyncio.StreamReader) -> dict[str, Any]:
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.decode().split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":")[1])
    body = await reader.readexactly(length) if length else b""
    return json.loads(body) if body else {}


def _response(obj: Any, status: str = "200 OK", chunked: bool = False) -> bytes:
    body = json.dumps(obj).encode()
    if chunked:
        mid = len(body) // 2
        parts = [body[:mid], body[mid:]]
        payload = b"".join(f"{len(p):x}\r\n".encode() + p + b"\r\n" for p in parts) + b"0\r\n\r\n"
        return f"HTTP/1.1 {status}\r\nTransfer-Encoding: chunked\r\n\r\n".encode() + payload
    return f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


async def _serve(handler: Handler) -> tuple[asyncio.Server, str]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


def _chat(content: dict[str, Any]) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": json.dumps(content)}}


@pytest.mark.parametrize("chunked", [False, True])
def test_round_trip(chunked: bool) -> None:
    seen: dict[str, Any] = {}

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        seen.update(await _read_request(r))
        w.write(_response(_chat({"ok": 1}), chunked=chunked))
        await w.drain()
        w.close()

    async def go() -> dict[str, Any]:
        server, url = await _serve(handler)
        async with server:
            llm = OllamaLLM("m", "d", url)
            return await llm.complete_json(system="s", user="u", schema={}, timeout_s=5)

    assert run(go()) == {"ok": 1}
    assert seen["model"] == "m" and seen["options"] == {"temperature": 0, "seed": 0}


def test_http_error_and_bad_body_raise_llm_error() -> None:
    replies = [_response({"error": "nope"}, "500 Internal Server Error"), _response({"x": 1})]

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _read_request(r)
        w.write(replies.pop(0))
        await w.drain()
        w.close()

    async def go() -> None:
        server, url = await _serve(handler)
        async with server:
            llm = OllamaLLM("m", "d", url)
            with pytest.raises(LLMError, match="HTTP 500"):
                await llm.complete_json(system="s", user="u", schema={}, timeout_s=5)
            with pytest.raises(LLMError, match="not JSON"):  # no message.content
                await llm.complete_json(system="s", user="u", schema={}, timeout_s=5)

    run(go())


def _hanging_server(closed: asyncio.Event) -> Handler:
    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _read_request(r)
        # Never answer; like Ollama, notice when the client goes away.
        if await r.read(1) == b"":
            closed.set()
        w.close()

    return handler


def test_cancellation_closes_the_connection() -> None:
    async def go() -> None:
        closed = asyncio.Event()
        server, url = await _serve(_hanging_server(closed))
        async with server:
            llm = OllamaLLM("m", "d", url)
            task = asyncio.create_task(
                llm.complete_json(system="s", user="u", schema={}, timeout_s=60)
            )
            await asyncio.sleep(0.2)  # request is in flight
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), timeout=2)  # server saw the disconnect

    run(go())


def test_timeout_raises_llm_error_and_closes_the_connection() -> None:
    async def go() -> None:
        closed = asyncio.Event()
        server, url = await _serve(_hanging_server(closed))
        async with server:
            llm = OllamaLLM("m", "d", url)
            with pytest.raises(LLMError, match="timed out"):
                await llm.complete_json(system="s", user="u", schema={}, timeout_s=0.3)
            await asyncio.wait_for(closed.wait(), timeout=2)

    run(go())


def test_connect_pins_digest_and_rejects_non_http() -> None:
    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _read_request(r)
        w.write(_response({"models": [{"name": "m", "digest": "abc"}]}))
        await w.drain()
        w.close()

    async def go() -> None:
        server, url = await _serve(handler)
        async with server:
            assert (await OllamaLLM.connect("m", base_url=url)).model_version == "ollama/m@abc"
            with pytest.raises(LLMError, match="pinned"):
                await OllamaLLM.connect("m", base_url=url, expected_digest="zzz")
        with pytest.raises(LLMError, match="plain http"):
            await OllamaLLM.connect("m", base_url="https://example.invalid")

    run(go())
