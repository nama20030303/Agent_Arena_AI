import { useMemo } from "react";
import { MapNode } from "../types";

interface Props {
  nodes: MapNode[];
  edges: { from: string; to: string }[];
  color: string;
  onSelect: (code: string) => void;
}

const W = 360;
const H = 560;

/** The knowledge constellation of one world: SVG prerequisite graph. */
export default function NodeGraph({ nodes, edges, color, onSelect }: Props) {
  const pos = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.code, { x: 30 + (n.x / 100) * (W - 60), y: 34 + (n.y / 100) * (H - 70) }));
    return m;
  }, [nodes]);

  const codes = useMemo(() => new Set(nodes.map((n) => n.code)), [nodes]);
  const byCode = useMemo(() => new Map(nodes.map((n) => [n.code, n])), [nodes]);

  return (
    <svg className="node-graph" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Карта знаний">
      <defs>
        <radialGradient id="ng-halo" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={color} stopOpacity="0.55" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </radialGradient>
      </defs>
      {edges
        .filter((e) => codes.has(e.from) && codes.has(e.to))
        .map((e, i) => {
          const a = pos.get(e.from)!;
          const b = pos.get(e.to)!;
          const from = byCode.get(e.from)!;
          const lit = from.status === "mastered";
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={lit ? color : "rgba(255,255,255,0.12)"}
              strokeOpacity={lit ? 0.6 : 1}
              strokeWidth={lit ? 1.6 : 1}
              strokeDasharray={lit ? undefined : "3 5"}
            />
          );
        })}
      {nodes.map((n) => {
        const p = pos.get(n.code)!;
        const r = n.is_boss ? 24 : 16;
        const locked = n.status === "locked";
        const mastered = n.status === "mastered";
        const available = n.status === "available";
        const active = n.status === "active";
        const circ = 2 * Math.PI * (r + 4);
        return (
          <g
            key={n.code}
            className={`gnode gnode-${n.status} ${n.is_boss ? "gnode-boss" : ""}`}
            transform={`translate(${p.x}, ${p.y})`}
            onClick={() => !locked && onSelect(n.code)}
            role="button"
            aria-label={`${n.name} — ${locked ? "закрыт" : mastered ? "освоен" : "доступен"}`}
            style={{ cursor: locked ? "default" : "pointer" }}
          >
            {(mastered || active || available) && <circle r={r * 2.2} fill="url(#ng-halo)" opacity={mastered ? 0.8 : 0.4} />}
            <circle
              r={r}
              fill={mastered ? color : locked ? "rgba(13,16,28,0.9)" : "rgba(20,26,46,0.95)"}
              stroke={locked ? "rgba(255,255,255,0.12)" : color}
              strokeOpacity={locked ? 1 : mastered ? 1 : 0.7}
              strokeWidth={n.is_boss ? 2.2 : 1.5}
            />
            {available && <circle className="pulse-ring" r={r + 5} fill="none" stroke={color} strokeWidth={1.4} />}
            {active && n.progress > 0 && (
              <circle
                r={r + 4}
                fill="none"
                stroke={color}
                strokeWidth={2}
                strokeLinecap="round"
                strokeDasharray={circ}
                strokeDashoffset={circ * (1 - n.progress / 100)}
                transform="rotate(-90)"
                opacity={0.9}
              />
            )}
            <text
              className="gnode-glyph"
              textAnchor="middle"
              dominantBaseline="central"
              fill={mastered ? "#061018" : locked ? "rgba(255,255,255,0.25)" : "#dbeafe"}
              fontSize={n.is_boss ? 15 : 11}
            >
              {locked ? "🔒" : n.is_boss ? "♛" : mastered ? "✓" : "◆"}
            </text>
            <text className="gnode-label" textAnchor="middle" y={r + 15} fill={locked ? "rgba(255,255,255,0.3)" : "rgba(226,236,255,0.85)"} fontSize={10}>
              {n.name.length > 24 ? n.name.slice(0, 23) + "…" : n.name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
