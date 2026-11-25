// Layered left-to-right layout for a plan DAG plus the nodes removed since the previous plan.
// Columns = topological layer. Within a column, nodes are ordered by their first consumer's
// position (processed from the output backwards), so each entity's branch stays together.

import type { PlanView } from "./api";

export const NODE_W = 196;
export const NODE_H = 46;
const COL_GAP = 84;
const ROW_GAP = 14;

export type Kind = "executed" | "reused" | "removed";

export interface LaidNode {
  id: string;
  operator: string;
  kind: Kind;
  x: number;
  y: number;
}

export interface LaidEdge {
  from: string;
  to: string;
  removed: boolean;
}

export interface Layout {
  nodes: LaidNode[];
  edges: LaidEdge[];
  width: number;
  height: number;
}

export function layoutPlan(plan: PlanView): Layout {
  type N = { id: string; operator: string; kind: Kind; layer: number };
  const all: N[] = [
    ...plan.nodes.map((n) => ({ id: n.id, operator: n.operator, kind: n.status, layer: n.layer })),
    ...plan.removed.map((n) => ({
      id: n.id,
      operator: n.operator,
      kind: "removed" as const,
      layer: n.layer,
    })),
  ];
  const ids = new Set(all.map((n) => n.id));

  // An edge between two removed nodes is reported by both ends; keep one copy.
  const edgeMap = new Map<string, LaidEdge>();
  const addEdge = (from: string, to: string, removed: boolean) => {
    if (ids.has(from) && ids.has(to)) edgeMap.set(`${from}->${to}`, { from, to, removed });
  };
  for (const n of plan.nodes) for (const i of n.inputs) addEdge(i, n.id, false);
  for (const r of plan.removed) {
    for (const i of r.inputs) addEdge(i, r.id, true);
    for (const c of r.consumed_by) addEdge(r.id, c, true);
  }
  const edges = [...edgeMap.values()];

  const consumers = new Map<string, string[]>();
  for (const e of edges) consumers.set(e.from, [...(consumers.get(e.from) ?? []), e.to]);

  const maxLayer = Math.max(0, ...all.map((n) => n.layer));
  const order = new Map<string, number>();
  const columns: N[][] = [];
  for (let layer = maxLayer; layer >= 0; layer--) {
    const col = all.filter((n) => n.layer === layer);
    const key = (n: N) => {
      const cs = (consumers.get(n.id) ?? []).map((c) => order.get(c) ?? Infinity);
      return cs.length ? Math.min(...cs) : Infinity;
    };
    col.sort((a, b) => key(a) - key(b) || a.id.localeCompare(b.id));
    col.forEach((n, i) => order.set(n.id, i));
    columns[layer] = col;
  }

  const tallest = Math.max(1, ...columns.map((c) => c?.length ?? 0));
  const height = tallest * (NODE_H + ROW_GAP) - ROW_GAP;
  const nodes: LaidNode[] = [];
  columns.forEach((col, layer) => {
    if (!col) return;
    const colH = col.length * (NODE_H + ROW_GAP) - ROW_GAP;
    const top = (height - colH) / 2;
    col.forEach((n, i) =>
      nodes.push({
        id: n.id,
        operator: n.operator,
        kind: n.kind,
        x: layer * (NODE_W + COL_GAP),
        y: top + i * (NODE_H + ROW_GAP),
      }),
    );
  });
  return { nodes, edges, width: (maxLayer + 1) * (NODE_W + COL_GAP) - COL_GAP, height };
}

export function prettyId(id: string): string {
  return id.split("__").join(" · ").replaceAll("_", " ");
}

// Graph label: the retriever suffix is redundant there (the operator is shown underneath).
export function nodeLabel(id: string): string {
  return prettyId(id.replace(/__(lex|dense)$/, ""));
}
