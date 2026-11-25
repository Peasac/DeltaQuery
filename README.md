# DeltaQuery

**Dependency-aware incremental retrieval for evolving questions.** A Python research
prototype that compiles a question into a DAG of retrieval operators, and when the user
revises the question ("replace Azure with GCP", "only RPO of zero", "also compare pricing")
reuses only the evidence that is provably still valid: correct under source updates,
permission revocations and freshness limits.

Status: research prototype. All results below are on a **synthetic** benchmark; nothing here
claims novelty or real-world performance. Full design and decision log: [`project.md`](project.md).

## Results (DeltaBench-synth, 300 revision sequences, 1,507 turns)

| Question | Result |
|---|---|
| Is reuse safe? | **Yes.** 0 turns cite stale or revoked evidence, versus 19–33% of turns for caches that ignore source versions or permissions. |
| Does it beat exact operator caching on cost? | **No (pre-stated go/no-go).** About 1.09× the retrieval ops of a version-aware exact cache; the difference is the work needed to *not* serve revoked documents. |
| Does it save work vs re-running everything? | Yes: 0.56× the retrieval ops of full re-execution. |
| Does evidence sufficiency help? | With adequate base retrieval: 29–40% less retrieval on constraint revisions at matched recall (held-out split, pre-registered selection). With weak base retrieval it trades recall for cost. |
| End to end with a local 4B LLM | Plan-equivalent intent match 0.65 on 40 providers; retrieval is ~0.1% of turn time. |

Reports: [`results/deltabench/`](results/deltabench/). Methodology: dev/test split by provider
family, repeated trials, bootstrap confidence intervals, ablations for each design choice,
and a planner change selected under a pre-registered rule.

## How it works

- **Content-addressed signatures** over operator, canonical parameters, input artifact
  hashes, snapshot/index/model versions and an access fingerprint. Equivalence is never
  inferred from embedding similarity.
- **Conservative reuse validation**: a signature hit only proposes reuse; a validator
  re-checks scope, versions, inputs, revocations and freshness. Fails closed on access.
- **Precise invalidation**: a revocation invalidates only nodes whose candidate domain
  contains the revoked source (checked against full re-execution in tests).
- **Evidence sufficiency**: deterministic claim extraction gates targeted retrieval;
  hedged statements never count as coverage.
- **LLM boundary**: the model only produces a schema-constrained intent and cited answer
  sentences; plans, edges and authorization never come from model output. Every answer
  sentence must cite verified source spans, or the system abstains.
- **Epochs**: a superseded request can never commit, even if its work ignores cancellation.

Stack: Python 3.11+, asyncio, Pydantic v2, SQLite, BM25 + dense retrieval, FastAPI, Next.js
trace UI, local models via Ollama. 101 tests, including mutation-checked safety properties.

## Commands

```
uv run pytest                                          # offline suite (no network)
uv run pytest -m live                                  # against local Ollama (qwen3.5:4b)
uv run python -m deltaquery.eval oracle --planner C    # benchmark, oracle-intent track
uv run python -m deltaquery.eval e2e --n 10            # end-to-end track (local LLM)
```

Trace UI (local development only; the API's X-Principal header is placeholder auth):

```
uv run python -m deltaquery.api --corpus tests/fixtures/corpus.json --grant alice:demo --entities aws,azure,gcp
cd web && npm install && npm run dev     # http://localhost:3000, proxies /api -> 127.0.0.1:8000
```

## Limitations

Synthetic data only (a real dataset, e.g. FinanceBench, is the next step); placeholder
authentication; the in-process cache is per process; cancelling a superseded request does
not abort its in-flight local LLM call.
