// Typed client for the DeltaQuery API (proxied at /api by next.config.mjs).
// Shapes mirror deltaquery/api/views.py.

export type Status = "executed" | "reused" | "skipped";

export interface EvidenceView {
  evidence_id: string;
  source_id: string;
  source_version: string;
  span: [number, number];
  score: number;
  coverage_type: string;
  quote: string | null;
}

export interface NodeView {
  id: string;
  operator: string;
  inputs: string[];
  status: Status;
  layer: number;
  signature: string;
  reuse_rejected: string | null;
  freshness_policy: string;
  max_age_s: number | null;
  params: Record<string, unknown>;
  evidence: EvidenceView[];
}

export interface RemovedNodeView {
  id: string;
  operator: string;
  inputs: string[];
  consumed_by: string[];
  layer: number;
}

export interface PlanView {
  session_id: string;
  version: number;
  output_id: string;
  nodes: NodeView[];
  removed: RemovedNodeView[];
}

export interface CitationView {
  ref: number;
  evidence_id: string;
  source_id: string;
  span: [number, number];
  quote: string;
}

export interface AnswerView {
  abstained: boolean;
  reason: string | null;
  sentences: { text: string; citations: CitationView[] }[];
  model_version: string | null;
}

export interface TraceDict {
  version: number;
  epoch: number;
  executed_nodes: string[];
  reused_nodes: string[];
  skipped_nodes: string[];
  removed_nodes: string[];
  reuse_rejections: [string, string][];
  timing_s: { compile: number; retrieval: number; generation: number };
}

export interface TurnView {
  session_id: string;
  version: number;
  trace_id: string;
  text: string;
  intent: {
    task: string;
    entities: string[];
    dimensions: string[];
    constraints: { attribute: string; op: string; value_seconds: number }[];
  };
  ops: { kind: string; old: string | null; new: string | null }[];
  answer: AnswerView | null;
  trace: TraceDict;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(principal: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", "X-Principal": principal },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  createSession: (principal: string, scope: string) =>
    call<{ session_id: string; scope: string; version: number }>(principal, "/sessions", {
      method: "POST",
      body: JSON.stringify({ scope }),
    }),
  ask: (principal: string, sid: string, text: string, expectedVersion: number) =>
    call<TurnView>(principal, `/sessions/${sid}/queries`, {
      method: "POST",
      body: JSON.stringify({ text, expected_version: expectedVersion }),
    }),
  trace: (principal: string, sid: string) =>
    call<{ session_id: string; scope: string; version: number; turns: TurnView[] }>(
      principal,
      `/sessions/${sid}/trace`,
    ),
  plan: (principal: string, sid: string, version: number) =>
    call<PlanView>(principal, `/sessions/${sid}/plans/${version}`),
};
