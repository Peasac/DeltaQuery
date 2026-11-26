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

**Phase 5 (DeltaBench, synthetic first)**
- Dataset `bench/deltabench-synth-v1/` (deterministic; manifest has seeds and SHA-256): 40 invented providers × 5 topics × 3 paragraphs; snapshot v1 changes DR and pricing facts for 12 providers; 300 sequences / 1,507 turns; dev/test split by provider family. Fact paragraphs are gold. Revision kinds: replace/add/remove entity, add/remove dimension, add/remove constraint, repeat, unrelated, and two world events (source update → v1; revocation of a document the current question uses) followed by re-asking. remove_* kinds are rare (13–16 each). Not human-validated.
- Oracle track (gold intents; differences are reuse only). Systems: deltaquery, full, exact_naive, exact_versioned (decisive), semantic@0.95, conv_hybrid; ablations abl_no_deps, abl_no_versions, abl_dq_keys (deltaquery's key design in an in-memory dict). Not implemented: tree baseline (no tree adapter), agentic repair, sufficiency ablation (no sufficiency feature). TTL events excluded: in a fully snapshot-versioned world TTL is redundant.
- Results, cold start, 3 trials (`results/deltabench/report-20261125T025159Z-87ed3d.md`): deltaquery uses 0.61× the retrieval ops of full re-execution but 1.19× [1.17, 1.22] those of exact_versioned. The entire excess is on revocation turns (policy version in every signature → whole-scope recompute); on every other kind op counts are identical. Stale-turn rate: deltaquery 0.000, exact_versioned 0.217, exact_naive 0.326, semantic 0.189. Recall equal across planner-based systems (0.844); conv_hybrid 0.282; abl_no_deps 0.216 (dependency propagation is necessary).
- Latency: deltaquery is 3.1× exact_versioned and 1.13× *slower than full re-execution*. abl_dq_keys (same keys, in-memory) is 1.08×, so the overhead is the SQLite store and thread hops, not the reuse logic. At this corpus size retrieval is sub-millisecond, so reuse cannot pay for any store access.
- Warm start (one store across sequences): ratio vs exact_versioned widens to 2.65× because each sequence's revocation yields a distinct policy version, isolating its artifacts. Stale: deltaquery 0.000, exact_versioned 0.251.
- Planner recall ceiling 0.844: with fused_k=3, the DR branch sometimes ranks the pricing fact paragraph (which mentions "disaster recovery") above the RPO paragraph. Same for all planner systems; data left unchanged to avoid tuning the benchmark to the system.
- **Go/no-go (§18): no-go for a cost claim.** DeltaQuery does not beat exact operator caching on cost; its cost equals a policy-aware exact cache (abl_dq_keys). What it does show is zero stale reuse where version- or policy-unaware caches serve stale or revoked evidence on 19–33% of turns. Per §18 the claim should narrow to safe reuse / evidence validity (and evidence sufficiency, not yet built). Two cost levers remain untested: precise revocation invalidation (only artifacts citing the revoked source) and an in-process cache in front of SQLite.
- E2E track (qwen3.5:4b, 10 sequences / 51 turns, single run; `results/deltabench/report-e2e-20261125T030020Z-a0fa90.md`): intent exact match 0.471, plan-equivalent match (ignoring `task`, which the planner does not use) 0.647, answer fact recall 0.604, citation precision 0.397 (against fact paragraphs only, so citing an overview paragraph counts as imprecise), abstain 0.059. Median per turn: compile 3.9 s, retrieval 0.013 s, generation 10.4 s, so retrieval reuse is irrelevant to end-to-end latency with a local LLM.
- Compiler failure modes: (1) neutral follow-ups ("Check again with the latest documents", "Show that again") flip `task` compare→describe (19/51 turns; plan unaffected); (2) on some neutral turns the model replaces the providers with the first entries of the 40-name entity enum (e.g. ardor, arthas, delmora, delwick): a degenerate constrained-decoding output, and it then propagates to later turns; (3) "RPO of zero" often also adds the disaster_recovery dimension (defensible reading; gold says no); add_constraint plan match is 0.125. The earlier 4/4 smoke result on the 3-provider fixture did not generalize to a 40-provider vocabulary. Mitigations to test: carry the prior intent forward unchanged when the model output only reorders the enum or the message contains no slot words, few-shot examples for neutral follow-ups, a larger model.

**Phase 5b (the two untested cost levers)**
- Lever 1, precise revocation invalidation: `Operator.access_dependencies` returns the revoked sources a node's own result can depend on. Retrievers: revoked ∩ their candidate domain (scope + entity filter), sound because BM25/dense scores of a passage do not depend on which *other* passages are excluded. Fusion/merge: none (inputs' hashes carry it). Default for any other operator: all revocations (conservative). The node signature carries a fingerprint of exactly that set; coarse mode (`precise_access=False`) reproduces the old scope-wide policy version exactly. Safety tests: incremental == full re-execution after revoking each of several sources, and after restoring a source that was revoked at compute time; both mutation-checked (a retriever that under-reports its dependencies fails them).
- Lever 2, in-process read-through artifact cache in `EvidenceStore` (safe: artifact rows are immutable; eviction purges entries) plus buffered usage writes (flushed in batches, before eviction, and on close). A validated in-memory copy survives later disk corruption; corruption is detected on the next cold load.
- Results, cold (`report-20261125T033551Z-e08d61.md`): retrieval ops vs exact_versioned 1.091× [1.080, 1.103] (was 1.191×); the residual is the re-run of the branch a revocation actually touches (2.44 ops/turn on revoke turns vs 5.10 coarse, 0 for exact_versioned, which serves the revoked document). Latency 2.36× exact_versioned (was 3.1×) and now 0.85× full re-execution (was 1.13×). Lever effects in isolation: precise access 0.92× ops; memcache 0.82× latency. abl_dq_keys (same keys, in-memory dict) is 1.04× exact_versioned latency, so the remaining latency gap is SQLite writes on execution.
- Warm (`report-20261125T034104Z-867d45.md`): ops 1.53× exact_versioned (was 2.65×), latency 1.78× (was 3.7×). exact_versioned's lower count here partly comes from reusing artifacts other sequences computed while a source was revoked, which lowers its recall (0.825 vs 0.844) on top of its 0.251 stale-turn rate.
- **Go/no-go unchanged: no-go for a cost claim.** With both levers DeltaQuery approaches but does not beat exact operator caching on cost, as expected: exact reuse is the floor for any signature-based system. Its distinct result remains safety (0 stale or revoked evidence vs 19–33% of turns for unaware caches) at a small, now-quantified cost premium. Next per §18: pivot to evidence sufficiency.

**Phase 6 (evidence sufficiency, the §18 pivot)**
- Claims: `operators/claims.py`, versioned regex extractors for typed exact values (`rpo=30s`, `rto=900s`). Hedged or approximate statements ("typically under one minute", "about thirty seconds", "of zero" with no unit) yield no claim; no claim means not covered, which means retrieve. No LLM is involved in the decision.
- Plan shape: for each constrained attribute, `{e}__{attr}__check` (`claim_check` over the entity's topic branches, no retrieval) plus the targeted retrieval branch gated by `Node.skip_if_nonempty`. Gated-off nodes get status `skipped` and an empty output. The gate's *decision*, not its content, is in the gated nodes' signatures, so a retrieval branch that stays needed across revisions is reused. (A first single-node design re-ran the targeted retrieval whenever topic branches changed; the Q4 test exposed it.)
- Safety: full == incremental with gates; revoking the covering source flips the gate and retrieval runs without it; the checks are mutation-tested (treat-anything-as-covering, never-covering, gate-ignored all fail tests). The §12 test "stricter constraint triggers retrieval when coverage is inadequate" now holds in both directions (adequate → skipped, inadequate → retrieved).
- Results, cold (`report-20261125T040221Z-f33dd6.md`), deltaquery vs dq_no_sufficiency (same runtime, always-retrieve planner): retrieval ops 0.974× [0.964, 0.982] overall and −15% on add_constraint turns (4.26 → 3.61), but evidence recall 0.844 → 0.830 overall and 1.000 → 0.934 on add_constraint turns. **Not matched quality**: it misses the ≤1 pp hypothesis overall. Cause: the targeted constraint retrieval was incidentally recovering gold passages the topic branches missed (e.g. the RTO paragraph), and only 3/40 providers' DR branch surfaces their exact-RPO paragraph (with fused_k=3 the pricing paragraph often outranks it). Sufficiency is only as good as the base branches' recall.
- Sufficiency is orthogonal to caching: every planner-based system running these plans gets the same savings (exact_versioned too), so it does not change the comparison with exact operator caching (deltaquery/exact_versioned ops 1.092×, stale 0.000 vs 0.217).
- Honest reading: as built and on this benchmark, sufficiency is a cost/quality trade-off, not a free win. Untested next steps: raise base-branch recall (DR template including RPO/RTO terms, larger fused_k) and re-measure both rates, being explicit that this is a post-hoc planner change; require coverage of every claim type a topic's gold implies before skipping (domain knowledge); evaluate on a real dataset.

**Phase 7 pre-registration (written before any run): planner v2 for base-branch recall**
- Motivation: Phase 6 showed sufficiency depends on base-branch recall. Changing templates after seeing results risks fitting the benchmark, so: candidates and rule are fixed here; selection uses the dev split (providers 0–19) only; the held-out test split (providers 20–39) is reported for the chosen variant.
- Candidates: A = current templates, fused_k=3 (control); B = DR query + "recovery point objective RPO recovery time objective RTO"; C = fused_k=5; D = B + C.
- Rule: maximize deltaquery evidence recall on dev (cold, 1 trial; recall and op counts are deterministic); if within 0.005 of the best, prefer fewer retrieval ops per turn. Report for the chosen variant vs A on test: recall, retrieval ops, and the sufficiency effect (deltaquery vs dq_no_sufficiency).

**Phase 7 results (planner v2, per the pre-registration above)**
- Dev split (selection), deltaquery recall / retrieval ops per turn: A 0.829 / 3.258, B 1.000 / 3.155, C 1.000 / 3.110, D 1.000 / 3.110. B, C and D tie on recall, so the rule prefers fewer ops; C and D then tie exactly (2,354 ops). **The pre-registered rule had no further tie-break**; C was chosen post hoc as the simpler single change. The rule also ignored evidence size: C and D return ~56% more passages per turn than A/B (13.1 vs 8.4), i.e. more generation tokens. B is therefore reported as a sensitivity analysis.
- Held-out test split, variant C (`report-20261125T041634Z-0fa559.md`, 3 trials): recall 1.000 with and without sufficiency; sufficiency cuts retrieval ops to 0.931× [0.911, 0.951] overall and from 4.18 to 2.50 per turn on add_constraint turns (−40%). Dev agrees (0.931×), so no sign of selection overfit. Variant B on test (`report-20261125T042537Z-3a1afd.md`): recall 1.000 both, ops 0.952× [0.934, 0.969], add_constraint 4.18 → 2.97, at A's evidence size.
- Control A on the same test split (`report-20261125T042613Z-3b87ef.md`): sufficiency recall 0.832 vs 0.847 without (add_constraint 0.939 vs 1.000) for 0.973× ops, i.e. the Phase 6 trade-off reproduces on held-out data, and the change between A and C is attributable to the planner fix.
- So with adequate base-branch recall, evidence sufficiency reduces retrieval at matched recall on this benchmark: the Phase 6 recall loss came from the base branches, not from the check. Caveats: (1) the synthetic benchmark is now saturated (every planner-based system reaches recall 1.000), so it cannot discriminate retrieval quality any further; (2) the planner change was made after seeing results (mitigated by pre-registration and the held-out split, not eliminated); (3) sufficiency is orthogonal to caching, so deltaquery vs exact_versioned is unchanged (1.09× ops, 0 vs 0.21 stale). A real dataset is now the binding constraint on any further claim.
- Default planner remains variant A in code (`PlannerConfig` defaults); variant C is selected via `--planner C` for benchmarks. Switching the library default is a separate decision.

**Phase 8 (compiler grounding), written before the live run**
- `compiler/grounding.py`: each change the LLM proposes relative to the prior intent is approved only if the message names it (entities by name; dimensions and constraint attributes via small keyword lists; constraint values must be stated). A message naming ≥2 new providers and none of the prior ones is a new question. Messages naming no slot carry the prior forward without a model call. `task` is derived from the entity count. A fresh question naming no provider fails with CompileError. Ungrounded changes are dropped (retrying at temperature 0 repeats the output). Unit fixtures reproduce each Phase 5 failure mode.
- Evaluation plan: compile-only track (`python -m deltaquery.eval compile --compare --n 40`), grounded vs ungrounded on the same 40 sequences (sequence seed 11), qwen3.5:4b, one run each; report exact and plan-equivalent match by revision kind, errors, and model-call rate. Known bias: the keyword lists match the benchmark's message templates by construction, so the gain partly reflects template overlap; real user phrasing will hit lexicon misses. Any later change to the lexicons must be evaluated on a fresh sequence seed, not these sequences.
- Results (`results/deltabench/report-compile-grounded-20261126T110749Z-2b2c62.md`; same 40 sequences / 206 turns, qwen3.5:4b, one run each): plan-equivalent match 0.709 → **0.961**, exact match 0.432 → 0.961, 0 compile errors either way; the model is called on 138/206 turns (neutral follow-ups skip it). Largest gains: add_constraint 0.118 → 0.941, the post-event "check again"/repeat turns 0.60–0.65 → 0.91–0.96, add_entity 0.786 → 1.000. Caveats: template overlap (see above); single run; synthetic phrasing.
- Remaining errors: all 8 trace to 2 first errors where, in a new question naming two providers, the model omitted one; grounding only removes ungrounded changes and never adds omitted ones, so the omission propagates. Candidate next rule: a provider named in a new question must be included. Per the rule above, it must be evaluated on a fresh sequence seed.

**Phase 9 (real data: FinanceBench feasibility)**
- Fetched only the open sample's two JSONL files (~1 MB, CC BY-NC 4.0; `data/` is gitignored, not redistributed; cite Islam et al. 2023, arXiv 2311.11944). No PDFs.
- Findings: 150 questions (50 metrics-generated, 50 domain-relevant, 50 novel-generated), 32 companies, 84 documents, 168 distinct gold evidence pages. **No metrics-generated template is shared by two companies**, so no "replace entity, same metric" revision pairs with gold for both turns exist in the open sample. 29 companies have ≥2 questions, but those are unrelated free-form questions (multi-step ratios), not slot edits, and they do not map onto closed topic templates. A corpus of only the provided evidence pages would be almost all gold.
- Conclusion: the open sample cannot support DeltaBench-style revision evaluation. Options: (1) request the full 10,231-question set (contact@patronus.ai), whose template questions likely recur across companies and years; (2) use the open sample only as a real-text retriever check (needs the PDFs for distractors; does not test revision/reuse); (3) a conversational dataset (QReCC / TREC CAsT), which requires an open-vocabulary compiler. Recommended: (1).

**Phase 10 pre-registration (written before the run): named providers in a new question**
- Change (`grounding-v2`): in a fresh question, every provider the message names is included even if the model omitted it (note "added named entity"). Revision turns are unchanged. Risk: a provider named in a negated sense ("not X") would be included; that gives a wrong plan, never stale reuse.
- Evaluation: compile track, `--compare`, fresh sequence seed **29** (never used before), 40 sequences, same world seed 7, qwen3.5:4b, one run each. Report plan-equivalent and exact match, errors and model-call rate. Success: grounded ≥ the Phase 8 grounded match (0.961) is *not* required; the question is whether grounding generalises to unseen sequences and whether the "omitted provider" error class disappears. Both outcomes get reported.

**Phase 11: domain configuration (behaviour-preserving refactor)**
- `deltaquery/domain.py`: a `Domain` lists topics (name, retrieval query, grounding keywords, optional TTL) and constrainable attributes (name, targeted query, grounding keywords, claim label regex). `CLOUD` reproduces the previous hardcoded values; it is the default everywhere.
- Intent topics/attributes are plain strings validated against the domain in the validation context; the LLM schema enums, compiler prompt, grounding keywords, planner queries/TTLs and claim-extractor labels all come from the domain. `Runtime.create(..., domain=...)` threads it through.
- The claim-check operator version includes a hash of the label patterns for any non-default domain, so coverage decided under one label set is never reused under another; the cloud default keeps its historical version.
- Verified unchanged for cloud: compiler prompt and JSON schema byte-identical to before; offline tests pass; oracle track (planner C, test split, 3 systems) identical turn for turn to `report-20261125T041634Z-0fa559` on every deterministic field (2,250 turn×system rows: ops, recall, stale, evidence counts, executed/reused node lists).
- Not yet general: constraint values are durations in seconds for every attribute; currency/percent units wait for a real dataset to say what is needed.

**Phase 10 results (fresh sequence seed 29, per the pre-registration above)**
- Report `results/deltabench/report-compile-grounded-20261126T164742Z-a61692.md`: 40 new sequences / 201 turns, qwen3.5:4b, one run each. Ungrounded: plan-equivalent 0.706, exact 0.463. Grounded (v2): **201/201 plan-equivalent and exact**, 0 errors; the model is called on 133/201 turns (68 neutral follow-ups carried forward).
- The ungrounded rate (0.706) matches the seed-11 run (0.709), so the new sequences are not easier for the model; grounding's gain held on unseen sequences.
- The v2 rule fired once, on exactly the targeted failure ("Compare Quindor and Ultris on disaster recovery": model omitted Quindor); without it, that first error would have propagated through the sequence. Other interventions: 13 unnamed dimensions and 9 unnamed entities dropped, 7 unrequested removals restored.
- What this does and does not show: it generalises across new provider combinations and sequence structures; it does **not** test new phrasing, because the fresh seed uses the same message templates the keyword lists were written against. 100% here is an upper bound for templated input, not an estimate for real users. Single run.

**Phase 12 pre-registration (written before any paraphrase is collected): new phrasing**
- Data: 15 conversations / 85 messages from sequence seed 29 (greedy selection covering every revision kind; `bench/paraphrase-v1/sheet.md`). A person rewrites every message from a description of its meaning (gold intent and diff), never seeing the benchmark wording or the keyword lists. Gold intents are unchanged.
- Run: `compile --compare --seq-seed 29 --texts bench/paraphrase-v1/sheet.md`, qwen3.5:4b, one run each. Compare with the same 15 conversations in the templated seed-29 run (`compile-*-20261126T16*`).
- Report: plan-equivalent match grounded vs ungrounded, by kind; every grounded miss classified as (a) lexicon miss (grounding blocked a correct model change, or carried forward a message that did change something), (b) model error, or (c) sheet ambiguity. No lexicon change before this report is written; any later lexicon fix is evaluated on a fresh paraphrase set, not this one.

**Phase 12 results (paraphrase track, per the pre-registration above)**
- Writer: no person was available, so ollama/qwen3.5:9b (a different, larger model than the 4B compiler; it never saw the benchmark wording or the keyword lists) filled the sheet from the meaning descriptions only (`bench/paraphrase-v1/sheet-qwen9b.md`). A first pass omitted the "currently:" state lines from the writer's input, and several re-asks referred to stale states; the input was fixed and all 15 were regenerated (not just the bad ones). Checking against the gold meanings only, 3 conversations were still excluded (re-asks naming the wrong providers; reasons and generated text kept in the file). 12 conversations / 67 messages remain.
- Report `results/deltabench/report-compile-grounded-20261126T193508Z-d8e088.md`. Plan-equivalent match on the same 67 turns:

  | | templated wording | paraphrased |
  |---|---:|---:|
  | ungrounded | 43/67 (0.642) | 41/67 (0.612) |
  | grounded | 67/67 (1.000) | **63/67 (0.940)** |

- All 4 grounded misses come from **one** first error, which then propagates: "oops, change that to Quinmora now" (current: Galnix and Isnix). The swap names only the new provider and refers to the old one as "that". The model itself proposed dropping *both* old providers; grounding correctly restored Isnix but also kept Galnix, because a removal must be named. Classification: (a) a grounding-rule miss (anaphoric replacement), compounded by (b) a model error. The failure is in the conservative direction (an extra provider gets retrieved; nothing stale or unrequested is dropped). Ungrounded, the same turn failed outright (invalid intent).
- No lexicon misses for topics or attributes: the writer varied verbs and structure but mostly kept topic words ("disaster recovery", "pricing"/"prices", "RPO"), so this set probes rephrased *actions* more than synonyms. It is a weak test of synonym coverage (e.g. "keys" for encryption, "outage recovery" for DR); a person-written set remains the stronger test. Single run; 67 turns, so the 0.940 has a wide interval.
- Candidate rule (not applied; must be evaluated on a fresh paraphrase set): a replacement verb ("change/switch/swap ... to X") whose old side is a pronoun may remove a prior entity when the model proposes exactly one removal. Today the model proposed two, so even this rule would need care.
