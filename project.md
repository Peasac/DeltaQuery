# DeltaQuery RAG — Project Spec

> Dependency-aware incremental retrieval over heterogeneous knowledge structures.
> Status: **proposed architecture. Novelty and performance are not yet established.**

## 1. Core Idea
Treat RAG as a **stateful execution engine**, not a fresh search per question. Compile each information request into a reusable retrieval plan (a DAG). When the user revises the question, diff the new plan against the old one, keep valid evidence, and recompute only the operations whose results are no longer valid.

## 2. Research Question
Can dependency-aware reuse across revised natural-language queries reduce retrieval and reasoning cost **at matched answer accuracy**, compared with:
1. full re-execution,
2. exact subquery (operator) caching,
3. semantic caching?

**Decisive test:** beat exact operator caching, not just naive RAG.

## 3. Deliverables
- Open-source Python retrieval runtime + SDK
- Multi-retriever adapter interface
- Versioned, immutable evidence store
- Interactive execution trace UI (executed / reused / invalidated / removed nodes)
- **DeltaBench**: benchmark of evolving-question sequences

## 4. Scope
| Tier | Content |
|---|---|
| Primary contribution | Incremental retrieval-plan reconciliation and safe evidence reuse |
| MVP | Text conversations; lexical (BM25), dense, document-tree retrieval |
| Extension | Graph/SQL operators, source updates, real-time voice |
| Non-goals | Generic chatbot, supporting every retrieval engine, replacing PageIndex or any vector DB |

## 5. Worked Example (acceptance scenario)
Frozen corpus of fictional cloud-provider docs.

| Revision | Intent change | Expected behavior |
|---|---|---|
| Q1 | compare(AWS, Azure, feature=DR) | Execute A (AWS), B (Azure), D (compare), E (generate) |
| Q2 | replace(Azure, GCP) | Reuse A; remove B; execute C (GCP); recompute D, regenerate E |
| Q3 | add_constraint(RPO == 0) | Check evidence sufficiency; expand retrieval if coverage inadequate; recompute extracted claims |
| Q4 | add_dimension(pricing) | Add pricing nodes; keep compatible DR evidence; apply freshness policy to prices |

Rules: reusable evidence ≠ reusable final answer. Unrelated query → full execution fallback. Every answer cites exact source spans.

## 6. Architecture (8 layers)
1. **Intent compiler** — query + prior intent → typed entities, predicates, scope, output requirements
2. **Logical planner** — DAG of retrieval, extraction, filter, join, fusion, synthesis operators
3. **Reconciler** — match old/new nodes by canonical signature; propagate changes
4. **Executor** — async, concurrent independent nodes; cancel obsolete work; enforce budgets
5. **Retriever adapters** — BM25, dense, doc-tree (later graph, SQL) behind common contracts
6. **Evidence store** — immutable source spans, result manifests, dependency edges, versions
7. **Validator** — freshness, access scope, evidence coverage, claim support
8. **Answer generator** — cited answer from valid evidence; abstain if insufficient

Principles: single process first (SQLite/Postgres); Redis only for distributed state/cancellation; planner, executor and evidence model independent of UI and LLM vendor.

## 7. Evidence Contract (the hard part)
```
Node { id, operator, canonical_parameters, input_ids, source_snapshot,
       index_version, model_version, access_scope, freshness_policy, output_contract }
Evidence { source_id, source_version, span, content_hash, retrieval_node_id,
           score, extracted_claims, coverage_type, created_at }
```
- Signature = content-addressed hash over canonical params + input artifact hashes + relevant versions.
- **Never** declare equivalence from embedding similarity alone. A semantic matcher may *propose* reuse; conservative validation must *approve*.
- Deterministic operators: exact equivalence testable. LLM operators: evaluate statistically, never claim strict equality.

### Reuse rules
| Change | Action |
|---|---|
| Entity replacement | Remove obsolete branch; execute new branch; recompute dependent joins |
| Added constraint | Check sufficiency of existing evidence; expand retrieval if needed |
| Source/index update | Invalidate dependents or rerun on new snapshot |
| Permission change | Reauthorize every reused artifact; never share across tenants |
| Changed rank fusion | Recompute fusion over current candidate sets |
| Model/prompt change | Invalidate affected generated artifacts |

### Fallback policy
Invalid compiler output, unclear scope, unknown freshness, or unprovable sufficiency → rerun branch or whole plan. **Fail closed on access control.**

## 8. Algorithm Sketch
`revise(session, text)`: compile intent → build + validate DAG → for each topological layer, resolve nodes concurrently (signature lookup → validate → reuse or execute+store) → grounded generate → if epoch superseded return `superseded()`, else commit. Dependencies come from the plan, never from LLM prose. Cancel removed nodes; suppress late results from old epochs. Production needs transactions, timeouts, error handling.

Objective: minimize cost (latency, tokens, compute) subject to evidence validity and quality threshold. MVP uses a rule-based policy; learned reuse policy is a later experiment.

## 9. Tech Stack
| Component | Choice |
|---|---|
| Runtime | Python 3.11+, FastAPI, asyncio, Pydantic |
| Lexical / dense | BM25 + FAISS/Qdrant (OpenSearch optional) |
| Doc tree | Simple heading/page tree; PageIndex adapter later |
| Embeddings | BGE-M3 or smaller local model |
| Persistence | SQLite (MVP) → PostgreSQL; Redis optional |
| Generation | Hosted LLM with structured output; local optional |
| UI / tracing | Next.js, DAG visualization, OpenTelemetry |
| Testing | pytest, frozen corpus, deterministic operator fixtures |

## 10. Repository Layout
```
deltaquery/
  compiler/    # intent schema, plan builder
  runtime/     # DAG, scheduler, reconcile, cancellation
  operators/   # bm25, dense, tree, fusion, extraction
  evidence/    # artifact store, provenance, validity
  api/         # sessions, query, trace endpoints
  eval/        # benchmark, baselines, metrics
  web/         # plan diff + evidence inspection
  tests/       # correctness + security fixtures
```

## 11. API
```
POST /sessions
POST /sessions/{id}/queries   {text, expected_version}
GET  /sessions/{id}/plans/{version}
GET  /sessions/{id}/trace
GET  /evidence/{id}
```
Each query returns answer, citations, plan version, trace ID. Report reused vs executed node counts, and retrieval time separately from generation time.

SDK target:
```python
session = runtime.session(scope="demo_corpus")
first = await session.ask("Compare AWS and Azure DR")
nxt = await session.ask("Replace Azure with GCP")
print(nxt.trace.reused_nodes, nxt.trace.executed_nodes)
```

## 12. Build Phases
1. **Deterministic core** — frozen corpus with stable source IDs + version hashes; typed operators; DAG executor; hand-authored plans. Test: full vs incremental execution yield identical artifacts.
2. **Retrieval adapters** — BM25 + dense on a shared source-span representation; doc-tree adapter; fusion kept separate (depends explicitly on both candidate lists); versioned index snapshots.
3. **NL revisions** — structured-output intent compiler (entity replace / add constraint / change dimension), schema-validated; build subqueries from intent slots, not free LLM rewrites.
4. **Correctness & DX** — provenance, permissions, freshness, eviction, fallback, trace UI, Python SDK.

### Minimum acceptance tests
- Identical query reuses valid artifacts
- Unrelated query triggers new execution
- Changed entity invalidates only dependent branches
- Stricter constraint triggers retrieval when coverage is inadequate
- Changed source and revoked permission prevent stale reuse
- Late results from superseded requests cannot be committed

## 13. Evaluation (DeltaBench)
- 200–500 revision sequences with gold intent slots, evidence spans, expected answer changes; human-validate a subset; held-out split by topic/document family.
- **Baselines:** full re-execution (same planner), exact operator cache, semantic cache, conversational hybrid RAG, vectorless tree, agentic/local-repair.
- **Data sources:** QReCC / TREC CAsT, FinanceBench, HotpotQA / MuSiQue (not native revision benchmarks — construct and validate revision sequences).
- **Metrics:** answer accuracy, evidence recall, citation support, stale-reuse rate, retrieval ops, tokens, P50/P95 latency, storage overhead; split by revision type and cold/warm cache.
- **Ablations:** canonicalization, dependency propagation, sufficiency checks, source-version checks.
- **Protocol:** fix corpus, model, budgets, hardware, concurrency; repeated trials with uncertainty intervals; compare quality at matched cost and cost at matched quality.
- **Hypotheses (not results):** ≥25% lower retrieval latency on revision-heavy workloads, ≤1 pp accuracy loss, near-zero stale reuse.

## 14. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| False reuse | Conservative signatures, sufficiency checks, full-execution fallback |
| Planning overhead | Lightweight templates; measure compiler cost |
| Stale data | Snapshots, TTLs, version checks |
| Prompt injection | Retrieved text is data; validate tool calls; authorization outside the model |
| Cache growth | Tenant-scoped quotas and eviction |
| Gains are just caching | Go/no-go gate (below) |

## 15. Voice Extension (post-MVP, optional)
Streaming ASR partials → speculative branches, cancellation on revision, retained compatible evidence, claim gate holding unsupported factual speech. Extra constraints: partial transcripts/ASR uncertainty, strict deadlines, barge-in, audio can't be retracted. Not a prerequisite.

## 16. Prior Art to Verify Before Claiming Novelty
PageIndex, Self-RAG (2310.11511), Adaptive-RAG (2403.14403), GraphRAG (2404.16130), RAPTOR (2401.18059), PlanRAG / Doctor-RAG / LedgerRAG (verify exact papers), database incremental view maintenance, semantic caches. Underlying techniques are established; the contribution is their evaluated application to evolving NL retrieval plans with explicit sufficiency + provenance.

## 17. Schedule (6 weeks, indicative)
| Week | Goal |
|---|---|
| 1 | Corpus, baselines, typed DAG |
| 2 | Lexical/dense adapters, artifact signatures |
| 3 | Revision compiler, incremental execution |
| 4 | Tree adapter, validity checks, trace UI |
| 5 | Benchmark + baseline comparisons |
| 6 | Ablations, failure analysis, paper draft |

## 18. Go / No-Go
Proceed to a paper **only if** the engine beats exact operator caching on meaningful revision workloads with preserved quality. If gains are all plain caching, or compiler overhead dominates, narrow the claim or pivot to **evidence sufficiency** as the contribution.

## 19. Implementation Decisions Log
**Phase 1 (deterministic core)**
- Signature excludes the plan-local node id; includes operator version, freshness policy and output contract in addition to the §7 fields (conservative: more fields → fewer false hits).
- `Evidence.created_at` is bookkeeping, not identity: `evidence_id` hashes every other field. `artifact_hash` = hash(output contract, ordered evidence ids).
- BM25 canonical query = sorted set of tokens. Scoring depends only on that set, so differently phrased queries with the same token set are *provably* the same computation (exact, not similarity-based).
- A deterministic `merge` operator (dedup-concat in declared input order) stands in for the compare/join node until fusion (Phase 2) and extraction exist. It is not rank fusion.
- Reuse requires a signature hit *and* a validator pass (scope, snapshot, index/model version, inputs). Any mismatch re-executes.
- Store is insert-only (UPDATE blocked by triggers). Re-storing a signature with different content raises `DeterminismViolation`, so every `reuse=False` run on a populated store doubles as a determinism check. DELETE stays allowed for future eviction.
- Plans may not mix access scopes across edges; BM25 filters on exact scope equality (fail closed).

**Phase 2 (retrieval adapters)**
- Shared span format = `PassageSet` (corpus snapshot + chunker version). BM25 and dense index exactly these passages, so evidence for the same text is byte-identical in (source_id, source_version, span, content_hash). Scope/entity filtering lives there too (exact scope match, fail closed).
- Dense: vendor-neutral `Embedder` protocol; `model_version` is part of the index version and every dense signature. Exact cosine search (no FAISS yet): at fixture scale it is cheap and has no ANN approximation that could differ between full and incremental runs. FAISS/Qdrant later = new index version.
- Tests use `HashingEmbedder` (char-trigram feature hashing): deterministic, offline, **not semantic**; never report it as a dense-model result. A real model (BGE) is not wired in yet.
- Dense query canonicalization is only NFC + whitespace collapse (a real model may be case/punctuation sensitive). BM25 keeps the stronger token-set canonicalization because its scoring provably ignores order and case.
- Fusion = RRF operator over ≥2 explicit `ranked_passages` inputs, summed in declared input order; emits new `fused` evidence rows (score and retrieval_node_id differ) on the same spans. Rejects inputs whose spans disagree on source_version/content_hash.
- Versioned snapshots: the executor routes each node by (operator, source_snapshot, index_version, model_version); several snapshots may be registered at once. A plan naming an unregistered snapshot is rejected, never routed to the "closest" one. Invalidation is snapshot-level for now: any source change re-runs every retriever on the new snapshot (conservative; per-document invalidation is later work).
- Note for real embedders: GPU/batch nondeterminism would surface as `DeterminismViolation` on re-execution. If that happens, dense must be treated as a statistically-evaluated operator (spec §7), not silently tolerated.

**Phase 3 (NL revisions)**
- LLM access goes through one `StructuredLLM` protocol (`deltaquery/llm/`). Ollama client is stdlib HTTP (no SDK), temperature 0, seed 0, thinking off; `model_version` = `ollama/<name>@<digest>` so a re-pulled model is a different version.
- The LLM returns the **complete** updated intent (task, entities, dimensions, constraints) under a JSON schema with closed vocabularies (entity enum injected from the corpus). It does not emit edit ops: small models were more reliable at restating a full object than at choosing ops, and the revision type is then computed exactly by `diff_intents` (replace/add/remove entity, add/remove dimension or constraint, change task).
- The prompt contains only the user's message and the prior intent JSON, never retrieved text. Invalid output → one retry → `CompileError`; the session then executes nothing.
- Plans come from slot templates (`compiler/planner.py`): per entity × (dimension | constrained attribute) a bm25 + dense → rrf branch, an entity merge, and a final merge. Entities/dimensions/constraints are sorted, so "AWS vs Azure" and "Azure vs AWS" are the same plan.
- Reuse safety does not depend on the LLM: a wrong intent yields a wrong *plan* (a quality error, measurable in DeltaBench), never a stale *reuse*, because reuse is still exact-signature + validator.
- Constraints currently always add targeted retrieval for the constrained attribute (conservative). Evidence-sufficiency checks that could skip it are not implemented yet.
- Live smoke check of the §5 sequence (single run each, temperature 0; not an accuracy measurement): `qwen3.5:4b` produced 4/4 expected intents; `llama3.2:3b` produced 2/4 (added unrequested RPO/RTO constraints on "Replace Azure with GCP"; re-added Azure on "Also compare pricing"). Default model: qwen3.5:4b.

**Phase 4 (correctness & DX; trace UI pending)**
- Access: `AccessPolicy` (grants per principal, revoked sources per scope, version per scope). The executor checks the principal against every scope in the plan before any reuse or execution; the scope's policy version is in every signature; retrievers exclude revoked sources; the validator re-authorizes each reused artifact (principal scope, policy version, no revoked source cited); the executor asserts no revoked source in new artifacts and refuses buggy operators. Coarse on purpose: any revocation in a scope invalidates all of that scope's artifacts (extra recompute only). Known gap: BM25 IDF statistics still include revoked passages (scores, not content).
- Freshness: `ttl` policy with `max_age_s`, measured from last computation *or re-verification* (re-executing with identical content renews it). The planner puts pricing retrieval on a 24h TTL. Unknown freshness → recompute.
- Eviction: per-scope LRU quota. Safe by construction under content addressing: evicted nodes recompute to the same hash. Orphaned evidence is dropped. `artifact_usage` is the only mutable table (bookkeeping only).
- Provenance: `get_evidence(id)`, `lineage(signature)` (inputs first; evicted ancestors reported as missing). Answer citations carry evidence id, source version, span and the verified quote (content hash re-checked).
- Fallback: a stored artifact that fails its integrity check is treated as a miss and recomputed (trace reason `corrupt`). A compile failure executes nothing.
- Epochs: each `ask` cancels the in-flight one; a turn commits only if its epoch is still current at completion, even if the old work ignored cancellation. `expected_version` gives optimistic concurrency. Stored artifacts from superseded work are kept (content-addressed, valid); only session state is guarded.
- Answers: generated outside the DAG and never cached. Evidence is presented as delimited, quoted data. Each sentence must cite in-range evidence; otherwise retry once, then abstain. The generator receives a question rendered deterministically from the intent plus the latest message (a revision like "Replace Azure with GCP" is not a question by itself).
- Schema field order matters under constrained decoding: with `abstain` first, qwen3.5:4b abstained on clearly sufficient evidence (2/2 probes); with `sentences` first it answered with correct citations (live test passes). Prompt version `answer-v2`.
- Live (qwen3.5:4b, single runs): §5 Q1–Q2 via the SDK, both answered with valid citations; reuse observed on Q2. Observed retrieval-quality issue: the DR branch also surfaces pricing passages that mention replication (a planner/query-template matter for DeltaBench, not a runtime bug).

**Phase 4b (API + trace UI)**
- FastAPI app (`deltaquery/api`) implements §11. Authentication is a placeholder: the principal comes from an `X-Principal` header and the dev server binds to 127.0.0.1 only. Authorization is real: sessions are private to their creator (others get 404), every request re-checks the scope grant, and `GET /evidence/{id}` serves evidence only if the caller can read a scope referencing it and the source is not revoked there (404 otherwise, so absence and denial look the same).
- Trace UI (`web/`, Next.js): layered plan graph with executed / reused / removed-since-previous nodes, node inspector (operator, status, refusal reason, freshness, signature, params, verified evidence quotes), answer with citation chips that highlight the nodes carrying that evidence, per-turn revision ops and compile/retrieval/generation timings. The page proxies `/api` to the local API, so the browser never talks to it cross-origin.
- Found while testing live: (1) Enter bypassed the disabled Send button, letting a second question supersede the first; the textarea is now read-only and submits are blocked while a request is in flight. (2) The Next dev proxy's ~30s default timeout cut off a slow local-LLM request that the server then committed, so UI and server versions diverged; the proxy timeout is now 300s and the UI resyncs from `GET /trace` after any failed ask. (3) Cancelling a superseded request cannot abort its in-flight Ollama HTTP call (it runs in a worker thread), so the next request queues behind it in Ollama. Correctness is unaffected (the late result never commits), but latency is: real cancellation needs an async HTTP client. Not fixed yet.
- Observed: a second session in the same scope reused every artifact computed by an earlier session (shared store, same signatures), as designed.
