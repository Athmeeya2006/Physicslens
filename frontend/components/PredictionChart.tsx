"use client";

import { useEffect, useRef, useState } from "react";

// The law found from the video carried to its prediction (Halley's 2061 return, Apophis's 2029 flyby), alone and with
// the planets added from JPL, against JPL. The planets are a toggle with their provenance on the chart, so nobody
// mistakes what was added for what was discovered. JPL is the neutral reference (grey, dashed); the discovered law
// carries the model hue; line style and the legend, never colour alone, tell them apart.

export interface Prediction {
  provenance: string;
  x_label: string;
  y_label: string;
  t: number[];
  alone: number[];
  with_planets: number[];
  jpl: number[];
  rows: { case: string; days_from_jpl?: number; hours_from_jpl?: number; km?: number; date?: string }[];
}

function useWidth(): [React.RefObject<HTMLElement | null>, number] {
  const ref = useRef<HTMLElement>(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function PredictionChart({ p }: { p: Prediction }) {
  const [box, W] = useWidth();
  const [planets, setPlanets] = useState(true);
  const H = 240;
  const left = 64, right = 14, top = 14, bottom = 40;
  const series = [p.alone, p.jpl, ...(planets ? [p.with_planets] : [])];
  const ys = series.flat().filter(Number.isFinite);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const t0 = p.t[0], t1 = p.t[p.t.length - 1];
  const x = (v: number) => left + ((v - t0) / (t1 - t0 || 1)) * (W - left - right);
  const y = (v: number) => top + (1 - (v - lo) / (hi - lo || 1)) * (H - top - bottom);
  const path = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(p.t[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000).toLocaleString()}k` : v.toPrecision(3));
  const value = (r: Prediction["rows"][number]) =>
    r.days_from_jpl !== undefined ? `${r.days_from_jpl >= 0 ? "+" : ""}${r.days_from_jpl.toFixed(1)} days`
      : `${(r.hours_from_jpl ?? 0) >= 0 ? "+" : ""}${(r.hours_from_jpl ?? 0).toFixed(2)} h, ${Math.round(r.km ?? 0).toLocaleString()} km`;
  return (
    <figure ref={box as React.RefObject<HTMLElement>} className="study-chart">
      <label className="toggle-row">
        <input type="checkbox" checked={planets} onChange={(e) => setPlanets(e.target.checked)} />
        With the planets <span className="muted small">({p.provenance})</span>
      </label>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${p.y_label} against ${p.x_label}: the discovered law alone${planets ? ", with the planets" : ""}, and JPL`}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="axis" x1={left} x2={W - right} y1={y(v)} y2={y(v)} opacity={0.3} />
            <text className="tick" x={left - 6} y={y(v) + 4} textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        <line className="axis" x1={x(0)} x2={x(0)} y1={top} y2={H - bottom} opacity={0.6} />
        <text className="tick" x={x(0)} y={H - bottom + 16} textAnchor="middle">JPL</text>
        <text className="tick" x={left} y={H - 6}>{p.x_label}</text>
        <path d={path(p.jpl)} className="pred-jpl" />
        <path d={path(p.alone)} className="pred-alone" />
        {planets && <path d={path(p.with_planets)} className="pred-planets" />}
      </svg>
      <figcaption className="forecast-caption">
        <span>
          <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" className="pred-jpl" /></svg>JPL</span>
          <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" className="pred-alone" /></svg>discovered law alone</span>
          {planets && <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" className="pred-planets" /></svg>with the planets</span>}
        </span>
        <span>{p.y_label}</span>
      </figcaption>
      <ul className="small" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
        {p.rows.filter((r) => planets || !r.case.includes("planets")).map((r) => (
          <li key={r.case}>{r.case}: {r.date ? `${r.date} (JPL)` : value(r)}</li>
        ))}
      </ul>
    </figure>
  );
}
