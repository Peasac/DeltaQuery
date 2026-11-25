# DeltaQuery

Research prototype: dependency-aware incremental retrieval. See `project.md` (spec + decisions
log, §19) and `CLAUDE.md` (working rules). Performance claims require DeltaBench output; none
are made here yet.

## Commands

```
uv run pytest                  # offline suite (no network; scripted LLM outputs)
uv run pytest -m live          # against local Ollama (default model qwen3.5:4b; DQ_LIVE_MODELS=a,b)
uv run ruff check . && uv run mypy
```

Trace UI (local development only; the API's X-Principal header is placeholder auth):

```
uv run python -m deltaquery.api --corpus tests/fixtures/corpus.json --grant alice:demo --entities aws,azure,gcp
cd web && npm install && npm run dev     # http://localhost:3000, proxies /api -> 127.0.0.1:8000
```
