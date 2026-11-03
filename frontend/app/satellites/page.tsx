"use client";

import { useEffect, useState } from "react";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { OnThisPage } from "@/components/OnThisPage";
import { ProofNav } from "@/components/ProofNav";

// Satellites on real orbits (scripts/satellite_demo.py): the ISS against its public catalogue orbit, what one pass
// and the observer's clock can tell, and manoeuvre detection with a negative control. Precomputed, served static.

interface Check { clock_error_s: number; time_offset_s: number; time_offset_sigma_s: number; intrack_km: number;
  crosstrack_km: number; crosstrack_sigma_km: number; km_per_clock_second: number; residual_rms_arcsec: number }
interface Run { injected: string; dv_m_s: number; injected_at_utc: string | null; passes: number; manoeuvre: boolean | null; reason: string;
  window_right?: boolean; impulse?: { dv_m_s: number; dv_ric_m_s: number[]; heldout_chi2: number }; one_orbit: { heldout_chi2: number; bic: number } }
interface Sat {
  oem_span: [string, string]; gp_epoch: string; sigma_arcsec: number;
  catalogue_vs_nasa: { days_from_epoch: number[]; intrack_km: number[]; crosstrack_km: number[]; radial_km: number[]; within_1_day: number };
  identification: { name: string; norad: number; rms_arcsec: number; clock_offset_s: number; same_orbit?: string[] }[];
  single_pass: { truth_catalogue_behind_s: number; checks: Check[]; utc: string; points: number };
  iod: string[]; manoeuvres: Run[];
  tumble_synthetic: { true_period_s: number; period_s: number; sigma_s: number; strongest_peak_s: number; doubled: boolean };
  outcome: Outcome;
}

function DriftChart({ d }: { d: Sat["catalogue_vs_nasa"] }) {
  const W = 640, H = 240, L = 52, R = 10, T = 12, B = 40;
  const xs = d.days_from_epoch, x0 = Math.min(...xs), x1 = Math.max(...xs);
  const all = [...d.intrack_km, ...d.crosstrack_km];
  const y0 = Math.min(...all), y1 = Math.max(...all);
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  const line = (ys: number[]) => xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
  const step = Math.max(1, Math.ceil((y1 - y0) / 5));
  const yt: number[] = [];
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) yt.push(v);
  return (
    <figure className="oc-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="How far the ISS catalogue orbit is from NASA's trajectory, by days from the catalogue epoch">
        {yt.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="axis" opacity={0.2} /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text></g>)}
        {[-2, 0, 2, 4, 6, 8, 10, 12].filter((v) => v >= x0 && v <= x1).map((v) => <text key={v} x={X(v)} y={H - B + 16} textAnchor="middle" className="tick">{v}</text>)}
        <line x1={X(0)} x2={X(0)} y1={T} y2={H - B} className="axis" opacity={0.5} strokeDasharray="3 3" />
        <path d={line(d.intrack_km)} fill="none" stroke="var(--model)" strokeWidth={2} />
        <path d={line(d.crosstrack_km)} fill="none" stroke="var(--tie)" strokeWidth={2} strokeDasharray="5 4" />
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">days from the catalogue orbit&apos;s epoch</text>
        <text x={6} y={10} className="tick">km</text>
      </svg>
      <figcaption className="small">
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--model)" strokeWidth="2" /></svg>in-track</span>
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--tie)" strokeWidth="2" strokeDasharray="5 4" /></svg>cross-track</span>
      </figcaption>
    </figure>
  );
}

interface History {
  rule: { statement: string };
  data: { distinct_files: number; first: string; last: string };
  vs_nasa_records: { burns_covered: number; detected: number; recall_ci95: [number, number]; false_alarms: number; quiet_days: number;
    timing_error_h_median: number; burns: { event: string; tig_utc?: string; tig?: string; dv_ms: number | null; detected?: boolean }[] };
  dv_vs_nasa: { burns: string[]; nasa_dv_ms: number; estimated_dv_ms: number }[];
}

// The detector on the ISS's real record (scripts/satellite_history.py): every reboost NASA recorded, found or not.
function RealRecord() {
  const [h, setH] = useState<History | null>(null);
  useEffect(() => {
    fetch("/astro/satellites_history.json").then((r) => (r.ok ? r.json() : null)).then(setH).catch(() => setH(null));
  }, []);
  if (!h) return null;
  const v = h.vs_nasa_records;
  return (
    <section className="panel stack" aria-labelledby="sat-real">
      <h2 id="sat-real">The real record: every ISS reboost from 2021 to 2026</h2>
      <p className="case-number">{v.detected} of {v.burns_covered} reboosts NASA recorded were found (95%: {Math.round(v.recall_ci95[0] * 100)}% to
        {" "}{Math.round(v.recall_ci95[1] * 100)}%), with {v.false_alarms} false alarm in {v.quiet_days.toFixed(0)} quiet days.</p>
      <p className="small">From {h.data.distinct_files} archived NASA trajectory files ({h.data.first.slice(0, 10)} to {h.data.last.slice(0, 10)}).
        The rule, fixed before the announcements were collected: {h.rule.statement}</p>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Recorded burn</th><th>Size, NASA</th><th>Size, measured here</th></tr></thead>
          <tbody>
            {h.dv_vs_nasa.map((x) => (
              <tr key={x.burns.join("+")}>
                <td data-label="Burn">{x.burns.join(", ")}</td>
                <td data-label="NASA" className="num">{x.nasa_dv_ms.toFixed(2)} m/s</td>
                <td data-label="Measured" className="num">{x.estimated_dv_ms.toFixed(2)} m/s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function SatellitesPage() {
  const [d, setD] = useState<Sat | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/astro/satellites.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The satellite results could not be loaded (${(e as Error).message}).`));
  }, []);

  return (
    <main id="main">
      <ProofNav caseTitle="satellites" />
      <header className="page-head">
        <h1>Satellites</h1>
        <p className="tagline">
          Which object is that streak, is it where its public catalogue orbit says, and did it move? Tested on the International
          Space Station: its catalogue orbit (CelesTrak) against the trajectory NASA publishes, seen by eight observers around the world with
          camera-like errors.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <OnThisPage items={[{ id: "sat-out", label: "Catalogue check" }, { id: "sat-real", label: "Real reboosts" }, { id: "sat-one", label: "One pass" }, { id: "sat-man", label: "Manoeuvres" }, { id: "sat-tum", label: "Tumbling" }]} />
          <section className="panel stack" aria-labelledby="sat-out">
            <h2 id="sat-out">The ISS against its catalogue orbit</h2>
            <OutcomeCard o={d.outcome} />
            <DriftChart d={d.catalogue_vs_nasa} />
            <p className="small muted">The catalogue orbit is a fit made for one moment (its epoch, {d.gp_epoch.slice(0, 16).replace("T", " ")} UTC) and drifts from the
              real trajectory as it ages: within a day of its epoch it stays within {d.catalogue_vs_nasa.within_1_day.toFixed(1)} km along the track.</p>
          </section>

          <RealRecord />

          <section className="panel stack" aria-labelledby="sat-one">
            <h2 id="sat-one">One pass: what it is, and what your clock can tell</h2>
            <p className="small">A pass at {d.single_pass.utc.replace("T", " ")} UTC, {d.single_pass.points} positions at {d.sigma_arcsec} arcsec each. Best matches:</p>
            <ol className="small">
              {d.identification.map((x) => (
                <li key={x.norad}>{x.name} (NORAD {x.norad}): {x.rms_arcsec.toFixed(0)} arcsec rms
                  {x.same_orbit && x.same_orbit.length > 0 ? `; ${x.same_orbit.join(", ")} share this orbit (docked modules catalogued separately)` : ""}</li>
              ))}
            </ol>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Observer&apos;s clock error</th><th>Catalogue behind by</th><th>In-track</th><th>Cross-track</th></tr></thead>
                <tbody>
                  {d.single_pass.checks.map((c) => (
                    <tr key={c.clock_error_s}>
                      <td data-label="Clock error" className="num">{c.clock_error_s} s</td>
                      <td data-label="Behind by" className="num">{c.time_offset_s.toFixed(3)} ± {c.time_offset_sigma_s.toFixed(3)} s</td>
                      <td data-label="In-track" className="num">{c.intrack_km.toFixed(2)} km</td>
                      <td data-label="Cross-track" className="num">{c.crosstrack_km.toFixed(2)} ± {c.crosstrack_sigma_km.toFixed(2)} km</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small">NASA&apos;s trajectory says the catalogue is {d.single_pass.truth_catalogue_behind_s.toFixed(3)} s behind on this pass. With a
              perfect clock one pass measures that; every second of clock error adds {d.single_pass.checks[0].km_per_clock_second.toFixed(1)} km along the track,
              and one pass cannot tell the two apart. A GPS-synced clock (or filming one at the start and end) is what makes the
              measurement mean something.</p>
            <h3>The observations as IOD lines, ready to submit</h3>
            <pre className="small" style={{ overflowX: "auto" }}>{d.iod.join("\n")}</pre>
            <p className="small muted">Format: Interactive Orbit Determination (G. Lewis, 1998; clarified 2002). Station 9999 is a placeholder: use your own.</p>
          </section>

          <section className="panel stack" aria-labelledby="sat-man">
            <h2 id="sat-man">Did it manoeuvre?</h2>
            <p className="small">Two laws for several days of passes: one orbit (gravity with the Earth&apos;s flattening, and drag), or one orbit with an
              impulse in a gap between passes. The impulse is accepted only if, fitted on the earlier passes, it forecasts the later
              ones and wins by BIC. The orbit model (gravity with the Earth&apos;s J2, J3 and J4 about the true pole, and drag) still misses NASA&apos;s
              trajectory by more than the camera&apos;s noise over four days, so when one orbit cannot forecast at all, an impulse that forecasts at least
              four times better counts too, and the verdict says so. The push along the track is measured well; the radial and cross-track parts
              absorb the model&apos;s leftovers and are not reliable.</p>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Injected</th><th>Passes</th><th>Verdict</th><th>Found</th></tr></thead>
                <tbody>
                  {d.manoeuvres.map((r) => (
                    <tr key={r.injected}>
                      <td data-label="Injected">{r.injected}{r.injected_at_utc ? ` at ${r.injected_at_utc.replace("T", " ")}` : ""}</td>
                      <td data-label="Passes" className="num">{r.passes}</td>
                      <td data-label="Verdict">{r.reason}</td>
                      <td data-label="Found" className="num">{r.manoeuvre && r.impulse ? `${r.impulse.dv_ric_m_s[1].toFixed(2)} m/s along the track${r.window_right ? ", right gap" : ", wrong gap"}` : r.manoeuvre === null ? "unresolved" : r.dv_m_s ? "missed" : "nothing (correct)"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel stack" aria-labelledby="sat-tum">
            <h2 id="sat-tum">Tumbling (synthetic)</h2>
            <p className="small">A dead satellite tumbling shows as flashes. The periodogram&apos;s strongest peak was {d.tumble_synthetic.strongest_peak_s.toFixed(1)} s;
              {d.tumble_synthetic.doubled ? " folding at twice that is clearly tighter (two different flashes per turn), so" : ""} the turn is
              {" "}{d.tumble_synthetic.period_s.toFixed(1)} ± {d.tumble_synthetic.sigma_s.toFixed(1)} s (true {d.tumble_synthetic.true_period_s} s).</p>
          </section>
          <p className="small muted">Scope: one observer and one pass constrain an orbit weakly; the clock dominates for amateurs; no real
            recorded manoeuvre has been reproduced yet (NASA&apos;s current ISS ephemeris lists none, and past catalogue orbits need a
            Space-Track account), so detection is shown on impulses added to the real orbit; tumble periods here are synthetic. For
            classified objects it reports only what public catalogues and your own observations show. Data: NASA JSC ISS ephemeris,
            CelesTrak GP elements.</p>
        </>
      )}
    </main>
  );
}
