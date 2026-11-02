"use client";

import { useEffect, useState } from "react";
import { Tex } from "@/components/Equation";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { ProofNav } from "@/components/ProofNav";

// The Sun's rotation law from real SDO/HMI images (scripts/sun_demo.py), and region-return forecasts checked against
// NOAA's records. Everything is precomputed and served as static files.

interface LawRow { law: string; title: string; latex: string; coef: Record<string, number>; sigma: Record<string, number>;
  heldout_rms_deg: number; heldout_chi2: number; bic: number; n: number }
interface Rate { lat: number; sidereal: number; sidereal_sigma: number; synodic: number; n: number; days: number; period: string }
interface Ret {
  period: string; lat: number; noaa_number: number | null; error_days?: number; noaa_error_days?: number;
  forecast: { east_limb_return: { utc: string; sigma_days: number }; west_limb: { utc: string } };
  returned: { number: number; listed: string; at: string; east_limb_utc: string } | null;
}
interface Sun {
  periods: { start: string; why: string; frames: number; tracks: number }[];
  status: string; reason: string; winner: string; laws: LawRow[]; sigma_deg: number; rates: Rate[]; rate_scatter_deg_day: number;
  law_curve: { lat: number[]; omega: number[]; sigma: number[] };
  published: Record<string, { source: string; omega: number[] }>;
  earth_rate_deg_day: { from_sunpy_L0: Record<string, number>; measured_sidereal_minus_synodic: number };
  returns: Ret[];
  returns_summary: { forecast: number; came_back: number; median_error_days: number | null; noaa_median_error_days: number | null; noaa_compared: number;
    later_than_forecast?: number; noaa_later_than_forecast?: number };
  outcome: Outcome;
}

const PROFILE_STYLE: Record<string, string> = { howard: "6 3", snodgrass: "2 3", allen: "10 4" };

function RateChart({ d }: { d: Sun }) {
  const W = 640, H = 340, L = 56, R = 12, T = 14, B = 44;
  const x0 = 0, x1 = 40;
  const ys = d.rates.map((r) => r.sidereal);
  const y0 = Math.min(12.4, ...ys), y1 = Math.max(14.9, ...ys);
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const path = (lat: number[], om: number[]) => lat.map((a, i) => `${i ? "L" : "M"}${X(a).toFixed(1)},${Y(om[i]).toFixed(1)}`).join("");
  const c = d.law_curve;
  const band = [...c.lat.map((a, i) => `${i ? "L" : "M"}${X(a).toFixed(1)},${Y(c.omega[i] + 1.96 * c.sigma[i]).toFixed(1)}`),
    ...c.lat.slice().reverse().map((a, j) => { const i = c.lat.length - 1 - j; return `L${X(a).toFixed(1)},${Y(c.omega[i] - 1.96 * c.sigma[i]).toFixed(1)}`; })].join("") + "Z";
  const yt = [12.5, 13, 13.5, 14, 14.5].filter((v) => v >= y0 && v <= y1);
  return (
    <figure className="oc-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Sidereal rotation rate against latitude: each sunspot, the discovered law and published profiles">
        {yt.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="axis" opacity={0.2} />
          <text x={L - 6} y={Y(v) + 5} textAnchor="end" className="tick">{v}</text></g>)}
        {[0, 10, 20, 30, 40].map((v) => <text key={v} x={X(v)} y={H - B + 18} textAnchor="middle" className="tick">{v}°</text>)}
        <path d={band} fill="var(--model)" opacity={0.15} />
        {d.rates.map((r, i) => (
          <circle key={i} cx={X(Math.abs(r.lat))} cy={Y(r.sidereal)} r={2.6} className="oc-tra" opacity={0.75}>
            <title>{`${r.period}: latitude ${r.lat.toFixed(1)}°, ${r.sidereal.toFixed(2)} ± ${r.sidereal_sigma.toFixed(2)} °/day over ${r.days.toFixed(1)} days`}</title>
          </circle>
        ))}
        {Object.entries(d.published).map(([k, p]) => (
          <path key={k} d={path(c.lat, p.omega)} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray={PROFILE_STYLE[k]} />
        ))}
        <path d={path(c.lat, c.omega)} fill="none" stroke="var(--model)" strokeWidth={2.5} />
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">latitude (north and south folded together)</text>
        <text x={8} y={12} className="tick">sidereal rate, degrees per day</text>
      </svg>
      <figcaption className="small">
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--model)" strokeWidth="2.5" /></svg>law found here (95% band)</span>
        {Object.entries(d.published).map(([k, p]) => (
          <span key={k} className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--ink-2)" strokeWidth="1.5" strokeDasharray={PROFILE_STYLE[k]} /></svg>{p.source}</span>
        ))}
        <span className="key"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3" className="oc-tra" /></svg>one sunspot</span>
      </figcaption>
    </figure>
  );
}

export default function SunPage() {
  const [d, setD] = useState<Sun | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/astro/sun.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The solar results could not be loaded (${(e as Error).message}).`));
  }, []);
  const best = d ? Math.min(...d.laws.map((l) => l.bic)) : 0;
  const earth = d ? Object.values(d.earth_rate_deg_day.from_sunpy_L0) : [];
  // The regions that came back are the check; the ones that never did are listed, folded, below them.
  const back = d ? d.returns.filter((r) => r.returned) : [];
  const gone = d ? d.returns.filter((r) => !r.returned) : [];

  return (
    <main id="main">
      <ProofNav caseTitle="the Sun's rotation" />
      <header className="page-head">
        <h1>The Sun&apos;s rotation</h1>
        <p className="tagline">
          A law found from real pixels with a known answer: sunspots tracked across SDO/HMI images show the Sun turning faster
          at its equator than nearer its poles. The same law says when an active region will face the Earth, and when it will
          come back around the east limb, checked here against NOAA&apos;s region records.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <section className="panel stack" aria-labelledby="sun-out">
            <h2 id="sun-out">Rotation law</h2>
            <OutcomeCard o={d.outcome} />
            <p className="small muted">{d.rates.length} sunspots tracked over {d.periods.length} periods of 14 days ({d.periods.map((p) => p.start).join(", ")}),
              one image every 6 hours. {d.reason}</p>
            <RateChart d={d} />
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Law</th><th>Coefficients (°/day)</th><th>Forecast of the last quarter</th><th>BIC behind the best</th></tr></thead>
                <tbody>
                  {d.laws.map((l) => (
                    <tr key={l.law}>
                      <td data-label="Law">{l.title}<div className="law-cell"><Tex tex={l.latex} /></div></td>
                      <td data-label="Coefficients" className="num">{Object.entries(l.coef).map(([k, v]) => `${k} = ${v.toFixed(2)} ± ${l.sigma[k].toFixed(2)}`).join(", ")}</td>
                      <td data-label="Forecast" className="num">{l.heldout_rms_deg.toFixed(2)}° (χ² {l.heldout_chi2.toFixed(1)}{l.heldout_chi2 <= 4 ? ", passes" : ", fails"})</td>
                      <td data-label="BIC behind" className="num">{(l.bic - best).toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small">Sidereal and synodic: the Carrington frame turns at 14.1844°/day against the stars by definition, so a
              spot&apos;s sidereal rate is that plus its drift in Carrington longitude. Seen from the moving Earth the Sun turns more slowly
              (synodic). The difference is the Earth&apos;s own motion, computed for each date: {Math.min(...earth).toFixed(3)} to {Math.max(...earth).toFixed(3)} °/day
              over these periods (faster near perihelion in January), not a fixed 0.9856; the spots themselves show
              {" "}{d.earth_rate_deg_day.measured_sidereal_minus_synodic.toFixed(3)} °/day.</p>
            <p className="small muted">Spots are not perfect tracers of the gas around them: their own motions scatter the rates by
              {" "}{d.rate_scatter_deg_day.toFixed(2)}°/day about the law, treated as noise.</p>
          </section>

          <section className="panel stack" aria-labelledby="sun-ret">
            <h2 id="sun-ret">When regions come back</h2>
            <p className="small">For each region still on the disc at the end of a period: when it leaves at the west limb and when it
              should reappear on the east limb, half a rotation later. NOAA gives a returning region a new number; it counts as the same
              region when it reappears within {5}° of latitude and {15}° of the predicted Carrington longitude.</p>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Region</th><th>Forecast back on the east limb</th><th>Seen again (NOAA)</th><th>Miss</th><th>NOAA&apos;s own forecast</th></tr></thead>
                <tbody>
                  {back.map((r, i) => (
                    <tr key={i}>
                      <td data-label="Region">{r.noaa_number ? `AR ${r.noaa_number}` : "unnumbered"} ({r.lat >= 0 ? "N" : "S"}{Math.abs(r.lat).toFixed(0)}, {r.period})</td>
                      <td data-label="Forecast" className="num">{r.forecast.east_limb_return.utc.replace("T", " ")} ± {r.forecast.east_limb_return.sigma_days.toFixed(1)} d</td>
                      <td data-label="Seen again">{r.returned ? `AR ${r.returned.number}, ${r.returned.at} on ${r.returned.listed}` : "not seen (decayed, or too small to number)"}</td>
                      <td data-label="Miss" className="num">{r.error_days !== undefined ? `${r.error_days >= 0 ? "+" : ""}${r.error_days.toFixed(1)} d` : ""}</td>
                      <td data-label="NOAA" className="num">{r.noaa_error_days !== undefined ? `${r.noaa_error_days >= 0 ? "+" : ""}${r.noaa_error_days.toFixed(1)} d` : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {gone.length > 0 && (
              <details>
                <summary className="small">{gone.length} more regions that were not seen again (decayed, or too small for NOAA to number)</summary>
                <div className="table-wrap">
                  <table className="cards">
                    <thead><tr><th>Region</th><th>Forecast back on the east limb</th><th>Seen again (NOAA)</th><th>Miss</th><th>NOAA&apos;s own forecast</th></tr></thead>
                    <tbody>
                      {gone.map((r, i) => (
                        <tr key={i}>
                          <td data-label="Region">{r.noaa_number ? `AR ${r.noaa_number}` : "unnumbered"} ({r.lat >= 0 ? "N" : "S"}{Math.abs(r.lat).toFixed(0)}, {r.period})</td>
                          <td data-label="Forecast" className="num">{r.forecast.east_limb_return.utc.replace("T", " ")} ± {r.forecast.east_limb_return.sigma_days.toFixed(1)} d</td>
                          <td data-label="Seen again">{r.returned ? `AR ${r.returned.number}, ${r.returned.at} on ${r.returned.listed}` : "not seen (decayed, or too small to number)"}</td>
                          <td data-label="Miss" className="num">{r.error_days !== undefined ? `${r.error_days >= 0 ? "+" : ""}${r.error_days.toFixed(1)} d` : ""}</td>
                          <td data-label="NOAA" className="num">{r.noaa_error_days !== undefined ? `${r.noaa_error_days >= 0 ? "+" : ""}${r.noaa_error_days.toFixed(1)} d` : ""}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            <p className="small muted">Median miss {d.returns_summary.median_error_days?.toFixed(1) ?? "n/a"} days over {d.returns_summary.came_back} returns
              {d.returns_summary.noaa_median_error_days != null ? `; NOAA's own "due to return" lists, ${d.returns_summary.noaa_median_error_days.toFixed(1)} days over ${d.returns_summary.noaa_compared}` : ""}.
              A region is listed when NOAA first numbers it, usually a day or two after it rounds the limb; the time it crossed the limb
              is worked back from where it was first listed.{d.returns_summary.later_than_forecast != null &&
              ` All misses go one way: ${d.returns_summary.later_than_forecast} of ${d.returns_summary.came_back} regions came back later than forecast here, and ${d.returns_summary.noaa_later_than_forecast} of ${d.returns_summary.noaa_compared} later than NOAA's own date.`}</p>
          </section>
          <p className="small muted">What it can&apos;t do: this is not new science (the law has been known for a century); sunspots do not
            rotate exactly like the plasma around them; and whether a region will flare is out of scope, only where it will be. Data:
            SDO/HMI continuum images via Helioviewer (NASA SDO, the HMI science team); NOAA SWPC Solar Region Summaries via NOAA NCEI.
            Positions from sunpy using each image&apos;s own header.</p>
        </>
      )}
    </main>
  );
}
