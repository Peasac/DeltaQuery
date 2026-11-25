"use client";

import { FormEvent, useMemo, useState } from "react";
import { PlanGraph } from "@/components/PlanGraph";
import { AnswerPanel, NodeInspector, Timing, TurnList } from "@/components/Panels";
import { api, type PlanView, type TurnView } from "@/lib/api";

export default function TracePage() {
  const [principal, setPrincipal] = useState("alice");
  const [scope, setScope] = useState("demo");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<TurnView[]>([]);
  const [version, setVersion] = useState<number | null>(null);
  const [plan, setPlan] = useState<PlanView | null>(null);
  const [nodeId, setNodeId] = useState<string | null>(null);
  const [citation, setCitation] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const turn = turns.find((t) => t.version === version) ?? null;
  const node = plan?.nodes.find((n) => n.id === nodeId) ?? null;
  const highlighted = useMemo(
    () =>
      new Set(
        citation && plan
          ? plan.nodes.filter((n) => n.evidence.some((e) => e.evidence_id === citation)).map((n) => n.id)
          : [],
      ),
    [citation, plan],
  );

  async function run<T>(fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function select(v: number, sid = sessionId) {
    if (!sid) return;
    setVersion(v);
    setNodeId(null);
    setCitation(null);
    const p = await run(() => api.plan(principal, sid, v));
    if (p) setPlan(p);
  }

  async function startSession(e: FormEvent) {
    e.preventDefault();
    const s = await run(() => api.createSession(principal, scope));
    if (!s) return;
    setSessionId(s.session_id);
    setTurns([]);
    setVersion(null);
    setPlan(null);
  }

  async function ask(e: FormEvent) {
    e.preventDefault();
    // One request at a time from this page (Enter must not bypass the disabled button).
    if (busy || !sessionId || !text.trim()) return;
    const question = text.trim();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const t = await api.ask(principal, sessionId, question, turns.length);
      setText("");
      setTurns((prev) => [...prev, t]);
      setBusy(false);
      await select(t.version, sessionId);
      return;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('"superseded"')) setNotice("A newer question superseded this one; it was not committed.");
      else setError(msg);
    }
    // The server is the source of truth: after any failure, resync committed turns.
    try {
      const tr = await api.trace(principal, sessionId);
      setTurns(tr.turns);
      if (tr.turns.length && tr.version !== turns.length) await select(tr.version, sessionId);
    } catch {
      /* keep the original error visible */
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <h1>DeltaQuery trace</h1>
        <form onSubmit={startSession} className="stack">
          <label>
            Principal <span className="muted small">(dev only, sent as X-Principal)</span>
            <input value={principal} onChange={(e) => setPrincipal(e.target.value)} disabled={!!sessionId} />
          </label>
          <label>
            Scope
            <input value={scope} onChange={(e) => setScope(e.target.value)} disabled={!!sessionId} />
          </label>
          {!sessionId ? (
            <button type="submit" disabled={busy}>
              Start session
            </button>
          ) : (
            <p className="muted small mono">session {sessionId}</p>
          )}
        </form>

        {sessionId && (
          <form onSubmit={ask} className="stack">
            <label>
              {turns.length ? "Revise the question" : "Ask a question"}
              <textarea
                rows={3}
                value={text}
                readOnly={busy}
                aria-busy={busy}
                placeholder={turns.length ? "e.g. Replace Azure with GCP" : "e.g. Compare AWS and Azure DR"}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) ask(e);
                }}
              />
            </label>
            <button type="submit" disabled={busy || !text.trim()}>
              {busy ? "Working…" : "Send"}
            </button>
          </form>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {notice && (
          <p className="notice" role="status">
            {notice}
          </p>
        )}
        <h2>Turns</h2>
        <TurnList turns={turns} selected={version} onSelect={(v) => select(v)} />
      </aside>

      <section className="main">
        {turn && plan ? (
          <>
            <header className="main-head">
              <div>
                <h2>
                  v{turn.version}: {turn.text}
                </h2>
                <p className="muted small">
                  {turn.intent.task} · {turn.intent.entities.join(", ")} · {turn.intent.dimensions.join(", ")}
                  {turn.intent.constraints.map((c) => ` · ${c.attribute} ${c.op} ${c.value_seconds}s`).join("")}
                </p>
              </div>
              <Timing turn={turn} />
            </header>
            <div className="legend">
              <span className="chip chip-executed">executed</span>
              <span className="chip chip-reused">reused</span>
              {plan.removed.length > 0 && (
                <span className="chip chip-removed">removed since v{turn.version - 1}</span>
              )}
              <span className="muted small">Click a node for its evidence; click a citation to find its source nodes.</span>
            </div>
            <PlanGraph plan={plan} selected={nodeId} highlighted={highlighted} onSelect={setNodeId} />
            <div className="bottom">
              <section className="card">
                <h3>Answer</h3>
                <AnswerPanel answer={turn.answer} active={citation} onCite={setCitation} />
              </section>
              <section className="card">
                <NodeInspector node={node} />
              </section>
            </div>
          </>
        ) : (
          <div className="empty">
            <p>Start a session and ask a question. Each revision shows which plan nodes were reused, executed, or removed.</p>
          </div>
        )}
      </section>
    </main>
  );
}
