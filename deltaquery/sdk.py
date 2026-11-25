"""Python SDK entry point.

    runtime = await Runtime.create(corpus=corpus, llm=llm, policy=policy)
    session = runtime.session(scope="demo", principal="alice")
    first = await session.ask("Compare AWS and Azure DR")
    nxt = await session.ask("Replace Azure with GCP")
    print(nxt.trace.reused_nodes, nxt.trace.executed_nodes)

One Runtime = one corpus snapshot, its indexes, one evidence store and one access policy.
Sessions share the store, so reuse also happens across sessions within the same scope (and
never across scopes: scope and access-policy version are part of every signature).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deltaquery.compiler.compiler import IntentCompiler
from deltaquery.compiler.planner import Planner, PlannerConfig
from deltaquery.evidence.access import AccessPolicy
from deltaquery.evidence.corpus import Corpus, PassageSet
from deltaquery.evidence.store import EvidenceStore
from deltaquery.llm.base import StructuredLLM
from deltaquery.operators.base import Operator
from deltaquery.operators.bm25 import BM25Index, BM25Operator
from deltaquery.operators.coverage import ClaimCheckOperator
from deltaquery.operators.dense import DenseIndex, DenseOperator
from deltaquery.operators.embedding import Embedder, HashingEmbedder
from deltaquery.operators.fusion import RRFOperator
from deltaquery.operators.merge import MergeOperator
from deltaquery.runtime.answer import AnswerGenerator, SpanResolver
from deltaquery.runtime.executor import Executor
from deltaquery.session import Session


@dataclass
class Runtime:
    corpus: Corpus
    passages: PassageSet
    bm25: BM25Index
    dense: DenseIndex
    store: EvidenceStore
    policy: AccessPolicy
    executor: Executor
    llm: StructuredLLM
    entities: frozenset[str]
    generate_answers: bool

    @classmethod
    async def create(
        cls,
        *,
        corpus: Corpus,
        llm: StructuredLLM,
        policy: AccessPolicy,
        store_path: str | Path = ":memory:",
        embedder: Embedder | None = None,
        entities: Iterable[str] | None = None,
        generate_answers: bool = True,
    ) -> Runtime:
        passages = PassageSet(corpus)
        bm25 = BM25Index(passages)
        dense = await DenseIndex.build(passages, embedder or HashingEmbedder())
        store = EvidenceStore(store_path)
        ops: list[Operator[Any]] = [
            BM25Operator(bm25),
            DenseOperator(dense),
            RRFOperator(),
            MergeOperator(),
            ClaimCheckOperator(passages),
        ]
        return cls(
            corpus=corpus,
            passages=passages,
            bm25=bm25,
            dense=dense,
            store=store,
            policy=policy,
            executor=Executor(store, ops, policy),
            llm=llm,
            entities=frozenset(
                entities if entities is not None else {d.entity for d in corpus.documents}
            ),
            generate_answers=generate_answers,
        )

    def session(self, *, scope: str, principal: str) -> Session:
        planner = Planner(PlannerConfig(bm25=self.bm25, dense=self.dense, access_scope=scope))
        generator = (
            AnswerGenerator(self.llm, SpanResolver(self.passages))
            if self.generate_answers
            else None
        )
        return Session(
            IntentCompiler(self.llm, self.entities),
            planner,
            self.executor,
            principal=principal,
            generator=generator,
        )

    async def close(self) -> None:
        await self.store.close()
