"use client";

import { formatUnit } from "@/lib/api";

import { useState } from "react";

// One metric per chart (never two y-scales). Colors validated on the light panel surface (#f5f6f3):
// series and flagged pass the lightness band, colorblind separation and contrast checks.
const SERIES = "#0b63ce";
const FLAGGED = "#b3261e";

export interface TrendPoint {
  label: string;
  value: number;
  std: number;
  flagged: boolean;
}

const W = 340;
const H = 170;
const PAD = { left: 52, right: 12, top: 22, bottom: 26 };

function niceTicks(lo: number, hi: number, n = 3): number[] {
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step);
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : a >= 0.01 ? v.toFixed(4) : v.toPrecision(2);
}

export function TrendChart({ title, unit, points }: { title: string; unit: string; points: TrendPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const lows = points.map((p) => p.value - p.std);
  const highs = points.map((p) => p.value + p.std);
  let lo = Math.min(...lows);
  let hi = Math.max(...highs);
  const pad = (hi - lo || Math.abs(hi) * 0.1 || 1) * 0.15;
  lo -= pad;
  hi += pad;
  const x = (i: number) => PAD.left + (points.length === 1 ? 0.5 : i / (points.length - 1)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.value)}`).join(" ");
  const h = hover !== null ? points[hover] : null;

  return (
    <figure className="trend-chart">
      <figcaption>{title} ({formatUnit(unit)})</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} across recordings`} onMouseLeave={() => setHover(null)}>
        {niceTicks(lo, hi).map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t) + 3} textAnchor="end" className="tick">{fmt(t)}</text>
          </g>
        ))}
        {points.map((p, i) => (
          <text key={`x${i}`} x={x(i)} y={H - 8} textAnchor="middle" className="tick">{p.label}</text>
        ))}
        <path d={path} fill="none" stroke={SERIES} strokeWidth={2} />
        {points.map((p, i) => {
          const color = p.flagged ? FLAGGED : SERIES;
          return (
            <g key={i}>
              <line x1={x(i)} x2={x(i)} y1={y(p.value - p.std)} y2={y(p.value + p.std)} stroke={color} strokeWidth={1.5} />
              <line x1={x(i) - 4} x2={x(i) + 4} y1={y(p.value - p.std)} y2={y(p.value - p.std)} stroke={color} strokeWidth={1.5} />
              <line x1={x(i) - 4} x2={x(i) + 4} y1={y(p.value + p.std)} y2={y(p.value + p.std)} stroke={color} strokeWidth={1.5} />
              {/* 2px surface ring keeps the marker readable where it overlaps the line */}
              <circle cx={x(i)} cy={y(p.value)} r={5} fill={color} stroke="#f5f6f3" strokeWidth={2} />
              {p.flagged && (
                <text x={x(i)} y={y(p.value + p.std) - 6} textAnchor="middle" className="flag-label">drift</text>
              )}
              {/* Hit target larger than the mark */}
              <rect x={x(i) - 14} y={PAD.top} width={28} height={H - PAD.top - PAD.bottom} fill="transparent"
                    onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} tabIndex={0} />
            </g>
          );
        })}
        {h && hover !== null && (
          <g className="tooltip" transform={`translate(${Math.min(x(hover) + 8, W - 120)},${PAD.top})`}>
            <rect width={112} height={38} rx={4} />
            <text x={8} y={15}>{h.label}{h.flagged ? " · drift" : ""}</text>
            <text x={8} y={30}>{fmt(h.value)} ± {fmt(h.std)}</text>
          </g>
        )}
      </svg>
    </figure>
  );
}
