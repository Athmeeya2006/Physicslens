"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { formatUnit, getSimToReal } from "@/lib/api";

// The sim-to-real gap as a picture: the real motion (grey dots, the reference) against the textbook model (amber,
// dashed) and the PhysicsLens model (blue, solid). Both models were fitted on the first three quarters only, so
// the shaded band is motion neither of them saw. Identity is carried by line style and the legend, not hue alone.

interface ModelErrors { heldout: number; whole: number; heldout_m?: number; whole_m?: number }

interface ControlStats {
  overshoot_pct: { mean: number; std: number };
  final_error_deg: { mean: number; std: number };
  abs_final_error_deg: { mean: number; std: number };
  settling_s: { mean: number; std: number } | null;
  settled: number;
  runs: number;
}

export interface SimToRealData {
  video: {
    family: string; label: string; unit: string; calibrated: boolean; heldout_start: number;
    textbook: { model_id: string; title: string; description: string; params: Record<string, number>; errors: ModelErrors };
    physicslens: { model_id: string; title: string; params: Record<string, number>; errors: ModelErrors };
    ratio_heldout: number; ratio_whole: number;
    t: number[]; measured: number[]; textbook_series: number[]; physicslens_series: number[];
  };
  control: {
    label: string; task: string; target_deg: number; settle_band_pct: number; horizon_s: number; plants: number;
    tuned_on: Record<"textbook" | "physicslens", { g: number; gamma: number; beta: number }>;
    summary: Record<"textbook" | "physicslens", ControlStats>;
  } | null;
  control_note?: string;
}

const PAD = { left: 52, right: 12, top: 26, bottom: 34 };

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

// Distance in the image plane when there is a scale; otherwise the coordinate itself (angles in degrees).
function errorText(e: ModelErrors, which: "heldout" | "whole", unit: string): string {
  const m = e[`${which}_m`];
  if (m != null) return `${(m * 1000).toFixed(m < 0.01 ? 2 : 1)} mm`;
  const v = e[which];
  if (unit === "rad") return `${((v * 180) / Math.PI).toFixed(2)}°`;
  return `${fmt(v)} ${unit === "m" ? "uncalibrated units" : formatUnit(unit)}`;
}

function OverlayChart({ v }: { v: SimToRealData["video"] }) {
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLElement>(null);
  const [W, setW] = useState(760);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = Math.round(Math.min(Math.max(W * 0.34, 200), 280));
  const { x, y, yTicks, xTicks, tbPath, plPath } = useMemo(() => {
    const all = [...v.measured, ...v.textbook_series, ...v.physicslens_series];
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.08;
    lo -= pad;
    hi += pad;
    const t0 = v.t[0];
    const t1 = v.t[v.t.length - 1];
    const x = (t: number) => PAD.left + ((t - t0) / (t1 - t0 || 1)) * (W - PAD.left - PAD.right);
    const y = (val: number) => PAD.top + (1 - (val - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
    const line = (s: number[]) => s.map((val, i) => `${i ? "L" : "M"}${x(v.t[i]).toFixed(1)},${y(val).toFixed(1)}`).join(" ");
    return { x, y, yTicks: ticks(lo, hi, 4), xTicks: ticks(t0, t1, Math.max(3, Math.floor(W / 130))),
             tbPath: line(v.textbook_series), plPath: line(v.physicslens_series) };
  }, [v, W, H]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    for (let i = 1; i < v.t.length; i++) if (Math.abs(x(v.t[i]) - px) < Math.abs(x(v.t[best]) - px)) best = i;
    setHover(best);
  };

  const unitText = !v.calibrated && v.unit === "m" ? "uncalibrated units" : formatUnit(v.unit);
  const bandX = x(v.heldout_start);
  const narrowBand = W - PAD.right - bandX < 250;
  const h = hover;
  const tipW = 190;

  return (
    <figure ref={box}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
           aria-label={`${v.label} over time: measured motion, the textbook model and the PhysicsLens model`}>
        <rect className="heldout-band" x={bandX} y={PAD.top} width={W - PAD.right - bandX} height={H - PAD.top - PAD.bottom} />
        <text className="heldout-label" x={narrowBand ? W - PAD.right : bandX + 8} y={PAD.top - 9}
              textAnchor={narrowBand ? "end" : "start"}>{W < 520 ? "Held out" : "Held out: neither model saw this"}</text>
        {yTicks.map((val) => (
          <g key={`y${val}`}>
            <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={y(val)} y2={y(val)} opacity={0.5} />
            <text className="tick" x={PAD.left - 8} y={y(val) + 4} textAnchor="end">{fmt(val)}</text>
          </g>
        ))}
        {xTicks.map((t) => <text key={`x${t}`} className="tick" x={x(t)} y={H - 12} textAnchor="middle">{fmt(t)} s</text>)}
        {v.measured.map((val, i) => <circle key={i} className="measured" cx={x(v.t[i])} cy={y(val)} r={2.1} opacity={0.75} />)}
        <path className="textbook-line" d={tbPath} />
        <path className="model-line" d={plPath} />
        {h !== null && (
          <g>
            <line className="cursor" x1={x(v.t[h])} x2={x(v.t[h])} y1={PAD.top} y2={H - PAD.bottom} />
            <g className="tip" transform={`translate(${x(v.t[h]) + 10 + tipW > W ? x(v.t[h]) - 10 - tipW : x(v.t[h]) + 10},${PAD.top + 6})`}>
              <rect width={tipW} height={68} rx={3} />
              <text x={10} y={18}>t = {fmt(v.t[h])} s{v.t[h] >= v.heldout_start ? ", held out" : ""}</text>
              <text x={10} y={34}>measured {fmt(v.measured[h])} {unitText}</text>
              <text x={10} y={48}>textbook {fmt(v.textbook_series[h])} {unitText}</text>
              <text x={10} y={62}>PhysicsLens {fmt(v.physicslens_series[h])} {unitText}</text>
            </g>
          </g>
        )}
      </svg>
      <figcaption className="forecast-caption">
        <span>
          <span className="key"><svg width="14" height="8" aria-hidden="true"><circle cx="7" cy="4" r="3" fill="#56616d" /></svg>Measured{unitText ? ` (${unitText})` : ""}</span>
          <span className="key"><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" stroke="#8f5b00" strokeWidth="2" strokeDasharray="6 4" /></svg>Textbook: {v.textbook.description}</span>
          <span className="key"><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" stroke="#0b63ce" strokeWidth="2" /></svg>PhysicsLens: {v.physicslens.title}</span>
        </span>
      </figcaption>
    </figure>
  );
}

function pm(s: { mean: number; std: number }, digits: number, unit: string): string {
  return `${s.mean.toFixed(digits)} ± ${s.std.toFixed(digits)}${unit}`;
}

export function SimToReal({ runId }: { runId: string }) {
  const [data, setData] = useState<SimToRealData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    getSimToReal(runId).then((d) => live && setData(d)).catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId]);

  if (error) return null; // the rest of the result stands on its own; this panel is extra evidence
  if (!data) {
    return (
      <div className="panel">
        <h3>Sim-to-real gap</h3>
        <p className="muted small">Simulating the textbook model and the identified model against the real motion…</p>
      </div>
    );
  }
  const v = data.video;
  const c = data.control;
  const gap = v.ratio_heldout >= 1.5
    ? `On the held-out quarter the textbook model lands ${v.ratio_heldout.toFixed(1)} times further from the real motion than the PhysicsLens model.`
    : `Here the textbook model does about as well (ratio ${v.ratio_heldout.toFixed(2)} on the held-out quarter): what PhysicsLens identified adds little to the textbook for this clip.`;
  const rows: ["textbook" | "physicslens", string][] = [["textbook", "Tuned on the textbook model"], ["physicslens", "Tuned on the PhysicsLens model"]];

  return (
    <div className="panel forecast">
      <h3>Sim-to-real gap: textbook physics against the identified model, on the real video</h3>
      <OverlayChart v={v} />
      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table>
          <thead><tr><th>Model</th><th>RMS error, held-out quarter</th><th>RMS error, whole clip</th></tr></thead>
          <tbody>
            <tr><td>Textbook: {v.textbook.description}</td>
              <td className="num">{errorText(v.textbook.errors, "heldout", v.unit)}</td><td className="num">{errorText(v.textbook.errors, "whole", v.unit)}</td></tr>
            <tr><td>PhysicsLens: {v.physicslens.title}</td>
              <td className="num">{errorText(v.physicslens.errors, "heldout", v.unit)}</td><td className="num">{errorText(v.physicslens.errors, "whole", v.unit)}</td></tr>
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: 8 }}>{gap}</p>
      <p className="muted small" style={{ marginTop: 4 }}>
        Both models were fitted on the first three quarters of the clip only (the textbook model gets its best starting
        state there), then compared with the measured motion.
      </p>

      {c && (
        <div style={{ marginTop: 20 }}>
          <h3>Control in simulation: a motor at the pivot</h3>
          <p className="sim-label" role="note">{c.label}</p>
          <p className="small" style={{ marginTop: 8 }}>{c.task} Each controller is run on {c.plants} pendulums drawn from the bootstrap refits of the PhysicsLens model.</p>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>Controller</th><th>Overshoot</th><th>Settling time (within {c.settle_band_pct.toFixed(0)}%)</th><th>Final amplitude error</th></tr></thead>
              <tbody>
                {rows.map(([k, name]) => {
                  const s = c.summary[k];
                  return (
                    <tr key={k}>
                      <td>{name}<div className="muted small">g = {c.tuned_on[k].g.toFixed(3)} m/s², damping {c.tuned_on[k].gamma || c.tuned_on[k].beta ? "included" : "none"}</div></td>
                      <td className="num">{pm(s.overshoot_pct, 1, "%")}</td>
                      <td className="num">{s.settling_s ? `${pm(s.settling_s, 2, " s")}` : "did not settle"}
                        {s.settled < s.runs && <div className="muted small">settled in {s.settled} of {s.runs} runs within {c.horizon_s.toFixed(0)} s</div>}</td>
                      <td className="num">{pm(s.final_error_deg, 2, "°")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            Mean ± standard deviation over the sampled pendulums, simulated in MuJoCo with the exported model plus a torque-limited
            motor. A negative final error means the swing stays short of the target.
          </p>
        </div>
      )}
    </div>
  );
}
