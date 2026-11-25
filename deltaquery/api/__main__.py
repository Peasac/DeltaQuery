"""Local dev server: python -m deltaquery.api --corpus PATH --grant alice:demo [...]

Binds to 127.0.0.1 only (authentication is a placeholder; see deltaquery.api.app).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from deltaquery.api.app import create_app
from deltaquery.evidence.access import AccessPolicy
from deltaquery.evidence.corpus import Corpus
from deltaquery.llm.ollama import OllamaLLM
from deltaquery.sdk import Runtime


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m deltaquery.api")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.5:4b", help="Ollama model name")
    ap.add_argument("--grant", action="append", default=[], metavar="PRINCIPAL:SCOPE")
    ap.add_argument("--entities", help="comma-separated entity vocabulary (default: all)")
    ap.add_argument("--store", default=":memory:", help="SQLite path for the evidence store")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    policy = AccessPolicy()
    for g in args.grant:
        principal, _, scope = g.partition(":")
        if not principal or not scope:
            ap.error(f"bad --grant {g!r}; expected PRINCIPAL:SCOPE")
        policy.grant(principal, scope)

    async def build() -> Runtime:
        llm = await OllamaLLM.connect(args.model)
        return await Runtime.create(
            corpus=Corpus.from_json(args.corpus),
            llm=llm,
            policy=policy,
            store_path=args.store,
            entities=args.entities.split(",") if args.entities else None,
        )

    runtime = asyncio.run(build())
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
