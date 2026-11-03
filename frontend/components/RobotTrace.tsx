"use client";

import { useMemo, useState } from "react";

// One held-out episode of the SO-100 arm (scripts/robot_demo.py, result.trace): the real joint angle against MuJoCo
// replays of the logged commands with the public model and with the identified one. The episode is the one with the
// median improvement, so the picture is typical, not the best case.

export interface Trace {
  episode: number; dt_s: number;
  joints: Record<string, { measured: number[]; command: number[]; public_model: number[]; identified: number[] }>;
}

const W = 760, H = 260, PAD = { l: 48, r: 12, t: 16, b: 30 };
const rms = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);

export function RobotTrace({ t }: { t: Trace }) {
  const names = Object.keys(t.joints);
  const [joint, setJoint] = useState(names.includes("shoulder_lift") ? "shoulder_lift" : names[0]);
  const j = t.joints[joint];
  // The error each model makes against the real arm, in degrees: on a 90-degree move a sub-degree miss is invisible
  // as two overlapping curves, but plain as distance from the zero line.
  const errPub = j.public_model.map((v, i) => v - j.measured[i]);
  const errId = j.identified.map((v, i) => v - j.measured[i]);
  const { path, yTicks, xTicks, zero } = useMemo(() => {
    const all = [...errPub, ...errId, 0];
    const m = Math.max(...all.map(Math.abs)) * 1.1 || 1;
    const lo = -m, hi = m;
    const n = j.measured.length, T = (n - 1) * t.dt_s;
    const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
    const path = (s: number[]) => s.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const step = m > 4 ? 2 : m > 2 ? 1 : m > 1 ? 0.5 : 0.25;
    const yTicks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) yTicks.push({ v: +v.toFixed(2), y: y(v) });
    const xTicks = [];
    const xs = T > 20 ? 5 : T > 8 ? 2 : 1;
    for (let s = 0; s <= T; s += xs) xTicks.push({ s, x: x(s / t.dt_s) });
    return { path, yTicks, xTicks, zero: y(0) };
  }, [errPub, errId, j.measured.length, t.dt_s]);
  const ePub = rms(j.public_model, j.measured), eId = rms(j.identified, j.measured);
  return (
    <figure className="robot-trace">
      <div className="chips" role="group" aria-label="Joint">
        {names.map((n) => (
          <button key={n} type="button" className={`chip${n === joint ? " on" : ""}`} aria-pressed={n === joint} onClick={() => setJoint(n)}>
            {n.replace("_", " ")}</button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${joint.replace("_", " ")} in held-out episode ${t.episode}: each model's error against the real arm`}>
        {yTicks.map((k) => (
          <g key={`y${k.v}`}>
            <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={k.y} y2={k.y} opacity={0.5} />
            <text className="tick" x={PAD.l - 6} y={k.y + 4} textAnchor="end">{k.v}°</text>
          </g>
        ))}
        {xTicks.map((k) => <text key={`x${k.s}`} className="tick" x={k.x} y={H - 10} textAnchor="middle">{k.s} s</text>)}
        <line className="rt-real" x1={PAD.l} x2={W - PAD.r} y1={zero} y2={zero} />
        <path d={path(errPub)} className="rt-public" />
        <path d={path(errId)} className="rt-ident" />
      </svg>
      <figcaption className="forecast-caption">
        <span>
          <span className="key"><span className="swatch line" style={{ background: "var(--ink)" }} />the real arm (zero error)</span>{" "}
          <span className="key"><span className="swatch line rt-dash" />public model: {ePub.toFixed(2)}° rms</span>{" "}
          <span className="key"><span className="swatch line model" />corrected model: {eId.toFixed(2)}° rms</span>
        </span>
        <span>Error against the real joint, degrees. Held-out episode {t.episode}, never used to fit: both models replay the same logged
          commands in MuJoCo.</span>
      </figcaption>
    </figure>
  );
}
