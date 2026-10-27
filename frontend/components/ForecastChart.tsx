"use client";

import { useMemo, useState } from "react";
import type { AnalysisResult } from "@/lib/api";

// The verification as a picture: measured motion (neutral reference dots) against the forecast of the model
// fitted on the first part only. The shaded band is data the model never saw. Measured is deliberately grey
// (the reference) and only the model carries hue; identity is also carried by shape (dots vs line) and the
// legend, never by colour alone. Model #0b63ce validated on the #f5f6f3 surface.

type Forecast = NonNullable<AnalysisResult["forecast"]>;

const W = 760;
const H = 260;
const PAD = { left: 56, right: 16, top: 26, bottom: 34 };

function ticks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}

export function ForecastChart({ f, uncalibrated = false }: { f: Forecast; uncalibrated?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const { x, y, yTicks, xTicks, path } = useMemo(() => {
    const all = [...f.measured, ...f.forecast];
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.08;
    lo -= pad;
    hi += pad;
    const t0 = f.t[0];
    const t1 = f.t[f.t.length - 1];
    const x = (t: number) => PAD.left + ((t - t0) / (t1 - t0 || 1)) * (W - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
    const path = f.forecast.map((v, i) => `${i ? "L" : "M"}${x(f.t[i]).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    return { x, y, yTicks: ticks(lo, hi, 4), xTicks: ticks(t0, t1, 6), path };
  }, [f]);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    for (let i = 1; i < f.t.length; i++) if (Math.abs(x(f.t[i]) - px) < Math.abs(x(f.t[best]) - px)) best = i;
    setHover(best);
  };

  const bandX = x(f.heldout_start);
  const h = hover;
  // Without a scale reference lengths are in arbitrary units; angles never need one.
  const unitText = uncalibrated && f.unit === "m" ? "uncalibrated units" : f.unit;
  const unit = unitText ? ` (${unitText})` : "";
  const narrowBand = W - PAD.right - bandX < 250;

  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" onMouseMove={onMove} onMouseLeave={() => setHover(null)}
           aria-label={`${f.label} over time: measured motion and the forecast of the ${f.title} fitted without the held-out part`}>
        <rect className="heldout-band" x={bandX} y={PAD.top} width={W - PAD.right - bandX} height={H - PAD.top - PAD.bottom} />
        <text className="heldout-label" x={narrowBand ? W - PAD.right : bandX + 8} y={PAD.top - 9}
              textAnchor={narrowBand ? "end" : "start"}>Held out: the model never saw this</text>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} opacity={0.5} />
            <text className="tick" x={PAD.left - 8} y={y(v) + 4} textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="tick" x={x(t)} y={H - 12} textAnchor="middle">{fmt(t)} s</text>
        ))}
        {f.measured.map((v, i) => (
          <circle key={i} className="measured" cx={x(f.t[i])} cy={y(v)} r={2.1} opacity={0.75} />
        ))}
        <path className="model-line" d={path} />
        {h !== null && (
          <g>
            <line className="cursor" x1={x(f.t[h])} x2={x(f.t[h])} y1={PAD.top} y2={H - PAD.bottom} />
            <g className="tip" transform={`translate(${Math.min(x(f.t[h]) + 10, W - 190)},${PAD.top + 6})`}>
              <rect width={180} height={52} rx={3} />
              <text x={10} y={18}>t = {fmt(f.t[h])} s{f.t[h] >= f.heldout_start ? ", held out" : ""}</text>
              <text x={10} y={34}>measured {fmt(f.measured[h])}{unitText ? ` ${unitText}` : ""}</text>
              <text x={10} y={48}>forecast {fmt(f.forecast[h])}{unitText ? ` ${unitText}` : ""}</text>
            </g>
          </g>
        )}
      </svg>
      <figcaption className="forecast-caption">
        <span>
          <span className="key"><svg width="14" height="8" aria-hidden="true"><circle cx="7" cy="4" r="3" fill="#56616d" /></svg>Measured{unit}</span>
          <span className="key"><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" stroke="#0b63ce" strokeWidth="2" /></svg>{f.title}, fitted on the first part only</span>
        </span>
        <span>Held-out forecast error: <strong className="num">{(f.heldout_nrmse * 100).toFixed(1)}%</strong> of the motion&apos;s spread (RMS over standard deviation)</span>
      </figcaption>
    </figure>
  );
}
