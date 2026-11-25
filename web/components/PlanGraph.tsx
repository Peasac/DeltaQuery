"use client";

import { useMemo } from "react";
import type { PlanView } from "@/lib/api";
import { layoutPlan, NODE_H, NODE_W, nodeLabel } from "@/lib/layout";

const PAD = 16;

export function PlanGraph({
  plan,
  selected,
  highlighted,
  onSelect,
}: {
  plan: PlanView;
  selected: string | null;
  highlighted: Set<string>;
  onSelect: (id: string) => void;
}) {
  const layout = useMemo(() => layoutPlan(plan), [plan]);
  const pos = new Map(layout.nodes.map((n) => [n.id, n]));

  return (
    <div className="graph-scroll">
      <svg
        width={layout.width + PAD * 2}
        height={layout.height + PAD * 2}
        role="img"
        aria-label={`Plan version ${plan.version}`}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="edge-head" />
          </marker>
        </defs>
        <g transform={`translate(${PAD},${PAD})`}>
          {layout.edges.map((e) => {
            const a = pos.get(e.from);
            const b = pos.get(e.to);
            if (!a || !b) return null;
            const x1 = a.x + NODE_W;
            const y1 = a.y + NODE_H / 2;
            const x2 = b.x - 2;
            const y2 = b.y + NODE_H / 2;
            const mx = (x1 + x2) / 2;
            return (
              <path
                key={`${e.from}->${e.to}`}
                d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                className={e.removed ? "edge edge-removed" : "edge"}
                markerEnd="url(#arrow)"
              />
            );
          })}
          {layout.nodes.map((n) => {
            const cls = [
              "node",
              `node-${n.kind}`,
              selected === n.id ? "node-selected" : "",
              highlighted.has(n.id) ? "node-highlight" : "",
            ].join(" ");
            const clickable = n.kind !== "removed";
            return (
              <g
                key={n.id}
                transform={`translate(${n.x},${n.y})`}
                className={cls}
                onClick={clickable ? () => onSelect(n.id) : undefined}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : undefined}
                onKeyDown={clickable ? (ev) => ev.key === "Enter" && onSelect(n.id) : undefined}
              >
                <title>{`${n.id} (${n.operator}) — ${n.kind}`}</title>
                <rect width={NODE_W} height={NODE_H} rx={8} />
                <text x={10} y={19} className="node-label">
                  {truncate(nodeLabel(n.id), 26)}
                </text>
                <text x={10} y={36} className="node-sub">
                  {n.operator}
                </text>
                <text x={NODE_W - 10} y={36} className="node-kind" textAnchor="end">
                  {n.kind}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
