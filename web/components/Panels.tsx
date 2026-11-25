"use client";

import type { AnswerView, NodeView, TurnView } from "@/lib/api";
import { prettyId } from "@/lib/layout";

export function TurnList({
  turns,
  selected,
  onSelect,
}: {
  turns: TurnView[];
  selected: number | null;
  onSelect: (v: number) => void;
}) {
  if (!turns.length) return <p className="muted">No questions yet.</p>;
  return (
    <ol className="turns">
      {turns.map((t) => (
        <li key={t.version}>
          <button
            className={`turn ${selected === t.version ? "turn-selected" : ""}`}
            onClick={() => onSelect(t.version)}
          >
            <span className="turn-head">
              <span className="turn-v">v{t.version}</span>
              <span className="turn-text">{t.text}</span>
            </span>
            <span className="chips">
              {t.ops.map((o, i) => (
                <span key={i} className="chip chip-op">
                  {o.kind.replaceAll("_", " ")}
                  {o.old || o.new ? `: ${[o.old, o.new].filter(Boolean).join(" → ")}` : ""}
                </span>
              ))}
            </span>
            <span className="counts">
              <span className="count count-executed">{t.trace.executed_nodes.length} executed</span>
              <span className="count count-reused">{t.trace.reused_nodes.length} reused</span>
              <span className="count count-removed">{t.trace.removed_nodes.length} removed</span>
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}

export function Timing({ turn }: { turn: TurnView }) {
  const ms = (s: number) => `${Math.round(s * 1000)} ms`;
  const t = turn.trace.timing_s;
  return (
    <dl className="timing">
      <div>
        <dt>Compile</dt>
        <dd>{ms(t.compile)}</dd>
      </div>
      <div>
        <dt>Retrieval</dt>
        <dd>{ms(t.retrieval)}</dd>
      </div>
      <div>
        <dt>Generation</dt>
        <dd>{ms(t.generation)}</dd>
      </div>
    </dl>
  );
}

export function AnswerPanel({
  answer,
  active,
  onCite,
}: {
  answer: AnswerView | null;
  active: string | null;
  onCite: (evidenceId: string | null) => void;
}) {
  if (!answer) return <p className="muted">Answer generation is off.</p>;
  if (answer.abstained)
    return (
      <p className="abstain">
        Abstained: <span>{answer.reason}</span>
      </p>
    );
  return (
    <div className="answer">
      {answer.sentences.map((s, i) => (
        <p key={i}>
          {s.text}{" "}
          {s.citations.map((c) => (
            <button
              key={c.ref}
              className={`cite ${active === c.evidence_id ? "cite-active" : ""}`}
              title={`${c.source_id} [${c.span[0]}–${c.span[1]}]\n${c.quote}`}
              onClick={() => onCite(active === c.evidence_id ? null : c.evidence_id)}
            >
              {c.ref}
            </button>
          ))}
        </p>
      ))}
      {answer.model_version && <p className="muted small">{answer.model_version}</p>}
    </div>
  );
}

export function NodeInspector({ node }: { node: NodeView | null }) {
  if (!node) return <p className="muted">Select a node in the plan to inspect its evidence.</p>;
  return (
    <div className="inspector">
      <h3>{prettyId(node.id)}</h3>
      <dl className="kv">
        <dt>Operator</dt>
        <dd>{node.operator}</dd>
        <dt>Status</dt>
        <dd>
          <span className={`chip chip-${node.status}`}>{node.status}</span>
          {node.reuse_rejected && (
            <span className="muted small"> (stored result refused: {node.reuse_rejected})</span>
          )}
        </dd>
        <dt>Freshness</dt>
        <dd>
          {node.freshness_policy}
          {node.max_age_s ? `, max age ${node.max_age_s}s` : ""}
        </dd>
        <dt>Signature</dt>
        <dd className="mono" title={node.signature}>
          {node.signature.slice(0, 16)}…
        </dd>
        {Object.keys(node.params).length > 0 && (
          <>
            <dt>Params</dt>
            <dd className="mono small">{JSON.stringify(node.params)}</dd>
          </>
        )}
        {node.inputs.length > 0 && (
          <>
            <dt>Inputs</dt>
            <dd>{node.inputs.map(prettyId).join(", ")}</dd>
          </>
        )}
      </dl>
      <h4>Evidence ({node.evidence.length})</h4>
      <ol className="evidence">
        {node.evidence.map((e) => (
          <li key={e.evidence_id}>
            <div className="ev-head">
              <span className="mono">{e.source_id}</span>
              <span className="muted small">
                [{e.span[0]}–{e.span[1]}] · {e.coverage_type} · {e.score.toFixed(3)}
              </span>
            </div>
            <blockquote>{e.quote ?? "(source text could not be verified)"}</blockquote>
          </li>
        ))}
      </ol>
    </div>
  );
}
