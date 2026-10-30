"use client";

import type { Depth3d } from "@/lib/api";
import { formatUnit, withUncertainty } from "@/lib/api";
import { Tex } from "./Equation";

// Depth over time from the ball's apparent size: measured depth with its per-frame error bar (grey, the
// reference) and the fitted 3D law's depth (the model colour), same visual language as the forecast check.

const W = 760;
const H = 240;
const PAD = { left: 52, right: 12, top: 14, bottom: 34 };

function ticks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const STATUS: Record<Depth3d["status"], string> = {
  accepted: "3D law verified",
  ambiguous: "3D laws tied",
  no_fit: "no 3D law fits",
  skipped: "not used",
};

export function DepthPanel({ d }: { d: Depth3d }) {
  if (d.status === "skipped" || !d.track?.length) {
    return (
      <div className="panel">
        <h3>Depth from one camera: {STATUS.skipped}</h3>
        <p className="muted small">{d.reason}</p>
      </div>
    );
  }
  const rows = d.track;
  const t = rows.map((r) => r[1]);
  const z = rows.map((r) => r[4]);
  const err = rows.map((r) => r[5]);
  const fitZ = d.fit_track?.map((p) => p[2]) ?? [];
  let lo = Math.min(...z.map((v, i) => v - err[i]), ...fitZ);
  let hi = Math.max(...z.map((v, i) => v + err[i]), ...fitZ);
  const pad = (hi - lo || 1) * 0.06;
  lo -= pad;
  hi += pad;
  const t0 = t[0];
  const t1 = t[t.length - 1];
  const x = (v: number) => PAD.left + ((v - t0) / (t1 - t0 || 1)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  const path = fitZ.map((v, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const g = d.g;
  const [gv, gs] = g ? withUncertainty(g.value, g.std) : ["", ""];
  const meanErr = err.reduce((a, b) => a + b, 0) / err.length;
  const meanZ = z.reduce((a, b) => a + b, 0) / z.length;

  return (
    <div className="panel forecast">
      <h3>Depth from one camera: {STATUS[d.status]}</h3>
      <p className="small">{d.reason}</p>
      <figure>
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Distance of the ball from the camera over time, measured from its apparent size, with error bars and the fitted 3D law">
          {ticks(lo, hi, 4).map((v) => (
            <g key={`y${v}`}>
              <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} opacity={0.5} />
              <text className="tick" x={PAD.left - 8} y={y(v) + 4} textAnchor="end">{v.toFixed(2)}</text>
            </g>
          ))}
          {ticks(t0, t1, 5).map((v) => (
            <text key={`x${v}`} className="tick" x={x(v)} y={H - 12} textAnchor="middle">{v.toFixed(2)} s</text>
          ))}
          {rows.map((r, i) => (
            <g key={i} opacity={r[7] ? 0.35 : 0.75}>
              <line style={{ stroke: "var(--ref)" }} strokeWidth={1} x1={x(t[i])} x2={x(t[i])} y1={y(z[i] - err[i])} y2={y(z[i] + err[i])} />
              <circle className="measured" cx={x(t[i])} cy={y(z[i])} r={2.1} />
            </g>
          ))}
          {path && <path className="model-line" d={path} />}
        </svg>
        <figcaption className="forecast-caption">
          <span>
            <span className="key"><svg width="14" height="10" aria-hidden="true"><line x1="7" x2="7" y1="0" y2="10" style={{ stroke: "var(--ref)" }} /><circle cx="7" cy="5" r="3" style={{ fill: "var(--ref)" }} /></svg>Distance from the camera (m), with its error</span>
            {path && <span className="key"><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" style={{ stroke: "var(--model)" }} strokeWidth="2" /></svg>{d.equation?.title}</span>}
          </span>
        </figcaption>
      </figure>
      {d.equation && <div className="equation"><Tex tex={d.equation.latex} display /></div>}
      {g && (
        <p className="num">
          <Tex tex="g" /> = {gv} ± {gs} {formatUnit(g.unit)} (95%: {g.ci95[0].toFixed(2)} to {g.ci95[1].toFixed(2)})
          {d.parameters?.filter((p) => p.name !== "g").map((p) => {
            const [v, s] = withUncertainty(p.value, p.std);
            return <span key={p.name}>, <Tex tex={p.latex} /> = {v} ± {s} {formatUnit(p.unit)}</span>;
          })}
        </p>
      )}
      <p className="muted small" style={{ marginTop: 8 }}>
        Depth comes from the ball&apos;s apparent size: one pixel of diameter error moves it by about {(100 * meanErr / meanZ).toFixed(1)}%
        here, so depth is far noisier than position across the image and is weighted accordingly in the fit. Used{" "}
        {d.frames_used} frames ({d.frames_interpolated ?? 0} bridged); rejected {d.frames_rejected?.blurred ?? 0} blurred and{" "}
        {d.frames_rejected?.occluded ?? 0} partly hidden. Focal length {d.focal_px?.toFixed(0)} px, ball diameter{" "}
        {((d.diameter_m ?? 0) * 100).toFixed(1)} cm (as entered; every length and g scale with it). Assumes the phone is
        upright and level, so gravity points straight down the image. Checked on synthetic throws only so far.
      </p>
    </div>
  );
}
