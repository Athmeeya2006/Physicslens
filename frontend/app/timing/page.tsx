"use client";

import { Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Tex } from "@/components/Equation";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { ProofNav } from "@/components/ProofNav";

// Transit timing on real data (scripts/timing_demo.py): WASP-12b's decaying orbit, Kepler-88's hidden planet, and
// which published ephemerides have gone stale. Everything is precomputed and served as static files.

type Point = [number, number, number, number, boolean]; // epoch, BJD, O-C min, sigma min, occultation
interface LawRow {
  law: string; title: string; latex: string; heldout_chi2: number; heldout_rms_min: number; bic: number;
  params: Record<string, number>; curve?: [number, number][];
}
interface Companion {
  candidate: { resonance: string; where: string; period: number }; Pc: number; Pc_sigma: number; mc_earth: number;
  mc_earth_sigma: number; eb: number; ec: number; heldout_rms_min: number; bic: number; curve?: [number, number][];
}
interface Timing {
  wasp12b: { system: string; status: string; reason: string; laws: LawRow[]; oc: Point[]; train_until_bjd: number;
    period_change_ms_per_year: [number, number]; next_windows: { epoch: number; utc_approx: string; sigma_min: number }[];
    outcome: Outcome; published: { source: string } };
  kepler88b: { system: string; status: string; reason: string; laws: LawRow[]; oc: Point[]; measured_ttv_period_days: number;
    measured_ttv_amplitude_h: number; transits_used: number; candidates: { resonance: string; where: string; period: number }[];
    companion_fits?: Companion[]; companion?: Companion; companion_degenerate_with?: Companion[]; outcome?: Outcome;
    published: { source: string; Pc_days: number; mc_earth: number } };
  rot: { date: string; source: string; planets: number; with_uncertainties: number; over_10_min: number; over_1_h: number;
    bright_over_1_h: number; top: { planet: string; sigma_min: number; period_days: number; vmag: number | null }[] };
}

const COLOURS = ["var(--model)", "var(--ok)", "var(--tie)", "#c084fc"];

/** Round tick values (1, 2 or 5 times a power of ten) covering lo to hi. */
function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo || 1;
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((v) => v >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(Math.abs(v) < 1e-9 * span ? 0 : v);
  return out;
}

function fmtTick(v: number): string {
  return Math.abs(v) >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : String(+v.toPrecision(3));
}

function OCChart({ oc, curves, cut, label }: { oc: Point[]; curves: { name: string; pts: [number, number][]; dashed?: boolean }[];
  cut?: number; label: string }) {
  const W = 640, H = 300, L = 56, R = 12, T = 28, B = 44;
  const xs = [...oc.map((p) => p[0]), ...curves.flatMap((c) => c.pts.map((p) => p[0]))];
  const ys = [...oc.map((p) => p[2]), ...curves.flatMap((c) => c.pts.map((p) => p[1]))];
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ylo = Math.min(...ys), yhi = Math.max(...ys);
  const pad = (yhi - ylo) * 0.08 || 1;
  const X = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - (ylo - pad)) / (yhi - ylo + 2 * pad || 1)) * (H - T - B);
  const cutEpoch = cut !== undefined ? oc.filter((p) => p[1] < cut).reduce((m, p) => Math.max(m, p[0]), -Infinity) : undefined;
  const ticks = niceTicks(ylo - pad, yhi + pad, 5);
  const xticks = niceTicks(x0, x1, 6).filter((v) => v >= x0 && v <= x1);
  return (
    <figure className="oc-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="axis" opacity={0.25} />
            <text x={L - 6} y={Y(v) + 5} textAnchor="end" className="tick">{fmtTick(v)}</text>
          </g>
        ))}
        {xticks.map((v) => (
          <text key={`x${v}`} x={X(v)} y={H - B + 18} textAnchor="middle" className="tick">{fmtTick(v)}</text>
        ))}
        {cutEpoch !== undefined && Number.isFinite(cutEpoch) && (
          <g>
            <rect x={X(cutEpoch)} y={T} width={W - R - X(cutEpoch)} height={H - T - B} className="heldout-band" />
            <text x={X(cutEpoch) + 4} y={T + 16} className="tick">held out: forecast, not fitted</text>
          </g>
        )}
        {curves.map((c, i) => (
          <path key={c.name} d={c.pts.map((p, k) => `${k ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}
                fill="none" stroke={COLOURS[i % COLOURS.length]} strokeWidth={2} strokeDasharray={c.dashed ? "5 4" : undefined} />
        ))}
        {oc.map((p) => (
          <g key={`${p[0]}-${p[4]}`}>
            <line x1={X(p[0])} x2={X(p[0])} y1={Y(p[2] - p[3])} y2={Y(p[2] + p[3])} className="axis" opacity={0.5} />
            <circle cx={X(p[0])} cy={Y(p[2])} r={2.4} className={p[4] ? "oc-occ" : "oc-tra"} />
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">epoch (orbits)</text>
        <text x={8} y={16} className="tick">O−C, minutes (observed minus a constant period)</text>
      </svg>
      <figcaption className="small">
        {curves.map((c, i) => (
          <span key={c.name} className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4"
            stroke={COLOURS[i % COLOURS.length]} strokeWidth={2} strokeDasharray={c.dashed ? "5 4" : undefined} /></svg>{c.name}</span>
        ))}
        <span className="key"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3" className="oc-tra" /></svg>transit</span>
        {oc.some((p) => p[4]) && <span className="key"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3" className="oc-occ" /></svg>occultation</span>}
      </figcaption>
    </figure>
  );
}

function LawTable({ laws }: { laws: LawRow[] }) {
  const best = Math.min(...laws.map((l) => l.bic));
  return (
    <div className="table-wrap">
      <table className="cards">
        <thead><tr><th>Law</th><th>Forecast of the later timings (χ² per event)</th><th>Typical miss</th><th>BIC behind the best</th></tr></thead>
        <tbody>
          {laws.map((l) => (
            <tr key={l.law}>
              <td data-label="Law">{l.title}<div className="law-cell"><Tex tex={l.latex} /></div></td>
              <td data-label="Forecast χ²" className="num">{l.heldout_chi2.toFixed(1)}{l.heldout_chi2 <= 4 ? " (passes)" : " (fails)"}</td>
              <td data-label="Typical miss" className="num">{l.heldout_rms_min.toFixed(1)} min</td>
              <td data-label="BIC behind" className="num">{(l.bic - best).toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TimingPage() {
  const [showAll, setShowAll] = useState(false);
  const [d, setD] = useState<Timing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [maxV, setMaxV] = useState(12);
  const [threshold, setThreshold] = useState(60);
  useEffect(() => {
    fetch("/astro/timing.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The timing results could not be loaded (${(e as Error).message}).`));
  }, []);
  const rot = useMemo(() => (d?.rot.top ?? []).filter((r) => r.vmag !== null && r.vmag <= maxV && r.sigma_min >= threshold),
    [d, maxV, threshold]);

  return (
    <main id="main">
      <ProofNav caseTitle="exoplanet timing" />
      <header className="page-head">
        <h1>Transit timing</h1>
        <p className="tagline">
          Telescopes schedule exoplanet observations from predicted transit times, and those predictions go stale. The
          same timings show physics: a transit arriving early or late can expose a planet nobody has seen, and a
          shrinking period means a planet spiralling into its star. All from public data; nothing here is rendered.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <section className="panel stack" aria-labelledby="w12">
            <h2 id="w12">WASP-12b: an orbit that is shrinking</h2>
            <OutcomeCard o={d.wasp12b.outcome} />
            <p className="small muted">158 published mid-times from 2008 to 2019 ({d.wasp12b.published.source}): each law was fitted
              on the earlier 75% and had to forecast the rest.</p>
            <OCChart oc={d.wasp12b.oc} cut={d.wasp12b.train_until_bjd} label="WASP-12b observed minus calculated transit and occultation times"
                     curves={d.wasp12b.laws.filter((l) => l.curve).map((l) => ({ name: l.title, pts: l.curve!, dashed: l.law !== "decay" }))} />
            <LawTable laws={d.wasp12b.laws} />
            <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
              <a className="button secondary" href="/astro/wasp12b_windows.ics" download><Download size={16} aria-hidden="true" /> Next transits (calendar)</a>
              <a className="button secondary" href="/astro/wasp12b_windows.csv" download><Download size={16} aria-hidden="true" /> Next transits (CSV)</a>
            </div>
            <ul className="small">
              {d.wasp12b.next_windows.slice(0, 5).map((w) => <li key={w.epoch}>{w.utc_approx.replace("T", " ")} UTC ± {w.sigma_min.toFixed(1)} min</li>)}
            </ul>
          </section>

          <section className="panel stack" aria-labelledby="k88">
            <h2 id="k88">Kepler-88: a planet found from the timing of another</h2>
            {d.kepler88b.outcome ? <OutcomeCard o={d.kepler88b.outcome} /> : (
              <p className="note">{d.kepler88b.reason}</p>
            )}
            <p className="small muted">
              {d.kepler88b.transits_used} transits of Kepler-88 b measured here from the Kepler light curves. They arrive up to
              {" "}{d.kepler88b.measured_ttv_amplitude_h.toFixed(0)} hours early or late, repeating every
              {" "}{d.kepler88b.measured_ttv_period_days.toFixed(0)} days. No simple law forecasts that; a second planet
              pulling on the first does. Its possible periods follow from the timing period near each resonance, and each
              was fitted with an N-body model and judged on the later transits.
            </p>
            <OCChart oc={d.kepler88b.oc} label="Kepler-88 b transit timing variations"
                     curves={[...(d.kepler88b.companion?.curve ? [{ name: "N-body: the hidden planet", pts: d.kepler88b.companion.curve }] : []),
                              ...d.kepler88b.laws.filter((l) => l.curve && l.law === "sinusoid").map((l) => ({ name: "one sinusoid", pts: l.curve!, dashed: true }))]} />
            <LawTable laws={d.kepler88b.laws} />
            {d.kepler88b.companion_fits && (
              <div className="table-wrap">
                <table className="cards">
                  <thead><tr><th>Companion candidate</th><th>Fitted period (d)</th><th>Mass (Earths)</th><th>Forecast of the later transits</th></tr></thead>
                  <tbody>
                    {d.kepler88b.companion_fits.map((c) => (
                      <tr key={`${c.candidate.resonance}-${c.candidate.where}-${c.candidate.period.toFixed(2)}`}>
                        <td data-label="Candidate">{c.candidate.resonance} resonance, {c.candidate.where}</td>
                        <td data-label="Period" className="num">{c.Pc.toFixed(3)} ± {c.Pc_sigma.toFixed(3)}</td>
                        <td data-label="Mass" className="num">{c.mc_earth.toFixed(0)} ± {c.mc_earth_sigma.toFixed(0)}</td>
                        <td data-label="Forecast" className="num">misses by {c.heldout_rms_min.toFixed(0)} min</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="small muted">Compared with {d.kepler88b.published.source}: period {d.kepler88b.published.Pc_days} days, mass
              {" "}{d.kepler88b.published.mc_earth} Earth masses, later confirmed by radial velocities. Mass and orbital eccentricity
              trade off in timing data, so masses are a range, not one number.</p>
          </section>

          <section className="panel stack" aria-labelledby="rot">
            <h2 id="rot">Ephemeris rot: which predictions have gone stale</h2>
            <p>
              On {d.rot.date}, {d.rot.over_10_min.toLocaleString()} of {d.rot.with_uncertainties.toLocaleString()} transiting
              planets in the NASA Exoplanet Archive will have transit times uncertain by more than 10 minutes, and
              {" "}{d.rot.over_1_h.toLocaleString()} by more than an hour ({d.rot.bright_over_1_h} of them around stars brighter
              than magnitude 12). These are the ones observers should time next.
            </p>
            <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
              <label className="small">Brighter than V
                <select value={maxV} onChange={(e) => setMaxV(Number(e.target.value))}>{[10, 11, 12, 13].map((v) => <option key={v} value={v}>{v}</option>)}</select>
              </label>
              <label className="small">Uncertainty above
                <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}>
                  <option value={10}>10 min</option><option value={60}>1 hour</option><option value={600}>10 hours</option>
                </select>
              </label>
            </div>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Planet</th><th>Transit-time uncertainty on {d.rot.date}</th><th>Period (d)</th><th>V</th></tr></thead>
                <tbody>
                  {rot.slice(0, showAll ? 60 : 15).map((r) => (
                    <tr key={r.planet}>
                      <td data-label="Planet">{r.planet}</td>
                      <td data-label="Uncertainty" className="num">{r.sigma_min >= 1440 ? `${(r.sigma_min / 1440).toFixed(1)} days` : r.sigma_min >= 60 ? `${(r.sigma_min / 60).toFixed(1)} h` : `${r.sigma_min.toFixed(0)} min`}</td>
                      <td data-label="Period" className="num">{r.period_days.toFixed(3)}</td>
                      <td data-label="V" className="num">{r.vmag?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {rot.length > 15 && (
              <p><button type="button" className="secondary" onClick={() => setShowAll((v) => !v)} aria-expanded={showAll}>
                {showAll ? "Show the 15 worst" : `Show ${Math.min(rot.length, 60) - 15} more`}</button></p>
            )}
            <p className="small muted">{d.rot.source}. Uncertainty = √(σ_T0² + (N σ_P)²) from the published ephemeris, N orbits ahead.</p>
          </section>
          <p className="small muted">What it can&apos;t do: timing inversions are degenerate (mass against eccentricity, and a single
            periodic signal cannot tell a companion from a turning orbit without more data), so PhysicsLens shows the families of
            solutions that fit, not one falsely precise answer.</p>
        </>
      )}
    </main>
  );
}
