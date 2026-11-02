"use client";

import { Download } from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { Tex } from "@/components/Equation";
import type { FallMapData } from "@/components/FallMap";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { AgentOnCase } from "@/components/AgentOnCase";
import { OnThisPage } from "@/components/OnThisPage";
import { ProofNav } from "@/components/ProofNav";

// A meteorite fall from its bright flight to the stones on the ground (scripts/fireball_demo.py on the Winchcombe
// fall of 2021). Everything is precomputed and served as static files.

const FallMap = dynamic(() => import("@/components/FallMap").then((m) => m.FallMap), { ssr: false });

interface Triage { verdict: "likely" | "possible" | "unlikely"; words: string; ln_alpha_sin_gamma: number; ln_beta: number }
interface Find {
  site: string; place: string; mass_g: string; lat: number; lon: number; along_m: number; across_m: number; includes?: string[];
  explained_by: "at_end" | "fragmentation_zone" | null;
  at_end: { along_range_m: [number, number]; across_offset_m: number };
  fragmentation_zone: { along_range_m: [number, number]; across_offset_m: number };
  blind?: { mass_bin_kg: number; hdr_level: number; inside_68: boolean; inside_95: boolean };
}
interface Fireball {
  event: string;
  dfn_example: { event: string; alpha: number; beta: number; objective: number; triage: Triage;
    notebook: { alpha: number; beta: number; objective: number; triage: string } };
  alpha_beta: { alpha: number; beta: number; alpha_sigma: number; beta_sigma: number; v0_ms: number; heldout_rms_kms: number;
    train_rms_kms: number; n: number; triage: Triage; entry_mass_kg: number; curve: [number, number][]; points: [number, number][] };
  single_body: { entry_mass_kg: number; heldout_rms_kms: number; fragmentation: { height_m: number; drag_jump: number; delta_bic: number } | null;
    atmosphere: string };
  orbit: { ours: Record<string, number>; published: Record<string, [number, number]>; pulls: Record<string, number> };
  dark_flight_vs_paper: { stone: string; shape: [number, number]; impact_ms: [number, number]; minutes: number;
    ours_impact_ms: number; ours_minutes: number }[];
  finds: Find[];
  fall_line: { mass_kg: number; lat: number; lon: number }[];
  strewn_points: { at_end: [number, number, number, number][]; fragmentation_zone: [number, number, number, number][] };
  end: { lat: number; lon: number; h_m: number };
  last_fragmentation_m: number;
  summary: { finds: number; median_across_m: number; max_across_m: number; along_at_end: number; along_either: number;
    blind_inside_68?: number; blind_inside_95?: number; blind_rule?: string };
  outcome: Outcome;
}

// Start of the bright flight (the published trajectory, first light at 90.6 km).
const FIRST_LIGHT: [number, number] = [51.870970, -3.109378];

const ORBIT_ROWS: [string, string, number][] = [["a_au", "Semi-major axis (AU)", 4], ["e", "Eccentricity", 4],
  ["i_deg", "Inclination (°)", 3], ["node_deg", "Ascending node (°)", 4], ["peri_deg", "Argument of perihelion (°)", 3],
  ["q_au", "Perihelion distance (AU)", 6], ["v_g_ms", "Speed before Earth's pull (m/s)", 0]];

/** The alpha-beta diagram: ln(alpha sin gamma) across, ln beta up, the two 50 g lines (Sansom et al. 2019). */
function AlphaBetaDiagram({ events }: { events: { name: string; t: Triage }[] }) {
  const W = 360, H = 300, L = 44, B = 36, T = 10, R = 10;
  const x0 = -1, x1 = 7, y0 = -3, y1 = 4;
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const line = (c: number, k: number) => {
    const pts: string[] = [];
    for (let x = x0; x <= x1; x += 0.02) {
      const inner = c - k * x;
      if (inner <= Math.exp(y0)) break;
      pts.push(`${pts.length ? "L" : "M"}${X(x).toFixed(1)},${Y(Math.min(Math.log(inner), y1)).toFixed(1)}`);
    }
    return pts.join("");
  };
  return (
    <figure className="ab-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Alpha-beta diagram: where each fireball sits against the two lines for a 50 gram meteorite">
        {[-2, 0, 2, 4].map((v) => <text key={`y${v}`} x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>)}
        {[0, 2, 4, 6].map((v) => <text key={`x${v}`} x={X(v)} y={H - B + 16} textAnchor="middle" className="tick">{v}</text>)}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} className="ab-frame" />
        <path d={line(13.2, 3)} className="ab-line mu0" />
        <path d={line(4.4, 1)} className="ab-line mu23" />
        <text x={X(0.2)} y={Y(-2.6)} className="tick">likely dropper</text>
        <text x={X(4.3)} y={Y(3.4)} className="tick">unlikely</text>
        {events.map((e, i) => (
          <g key={e.name}>
            <circle cx={X(e.t.ln_alpha_sin_gamma)} cy={Y(e.t.ln_beta)} r={5} className={`ab-point ab-${e.t.verdict}`} />
            <text x={X(e.t.ln_alpha_sin_gamma) + 8} y={Y(e.t.ln_beta) + 4 + i * 2} className="tick">{e.name}</text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">ln(α sin γ)</text>
        <text x={6} y={T + 12} className="tick">ln β</text>
      </svg>
      <figcaption className="small muted">Solid: no spin (μ = 0); dashed: spinning, ablating evenly (μ = 2/3). Below the dashed line a
        50 g meteorite is likely; above the solid line it is unlikely (ρ = 3500 kg/m³, c<sub>d</sub>A = 1.5).</figcaption>
    </figure>
  );
}

/** Speed against height: the measured points and the fitted alpha-beta law. */
function SpeedChart({ points, curve }: { points: [number, number][]; curve: [number, number][] }) {
  const W = 420, H = 260, L = 48, B = 36, T = 10, R = 10;
  const hs = points.map((p) => p[0]), vs = points.map((p) => p[1]);
  const h0 = Math.min(...hs), h1 = Math.max(...hs), v0 = Math.min(...vs) * 0.95, v1 = Math.max(...vs) * 1.03;
  const X = (v: number) => L + ((v - v0) / (v1 - v0)) * (W - L - R);
  const Y = (h: number) => T + (1 - (h - h0) / (h1 - h0)) * (H - T - B);
  const ticksV = [4, 6, 8, 10, 12, 14].map((k) => k * 1000).filter((v) => v >= v0 && v <= v1);
  const ticksH = [30, 40, 50, 60, 70, 80, 90].map((k) => k * 1000).filter((h) => h >= h0 && h <= h1);
  return (
    <figure className="ab-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Speed against height: measured from seven cameras, with the fitted alpha-beta law">
        {ticksH.map((h) => <text key={h} x={L - 6} y={Y(h) + 4} textAnchor="end" className="tick">{h / 1000}</text>)}
        {ticksV.map((v) => <text key={v} x={X(v)} y={H - B + 16} textAnchor="middle" className="tick">{v / 1000}</text>)}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} className="ab-frame" />
        {points.map(([h, v]) => <circle key={`${h}-${v}`} cx={X(v)} cy={Y(h)} r={2} className="oc-tra" />)}
        <path d={curve.filter(([, h]) => h >= h0 && h <= h1).map(([v, h], k) => `${k ? "L" : "M"}${X(v).toFixed(1)},${Y(h).toFixed(1)}`).join("")}
              className="ab-fit" />
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">speed (km/s)</text>
        <text x={6} y={T + 12} className="tick">km</text>
      </svg>
    </figure>
  );
}

export default function FireballPage() {
  const [d, setD] = useState<Fireball | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState<"at_end" | "fragmentation_zone">("at_end");
  useEffect(() => {
    fetch("/astro/fireball.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The fireball results could not be loaded (${(e as Error).message}).`));
  }, []);
  const map: FallMapData | null = useMemo(() => d && {
    track: [FIRST_LIGHT, [d.end.lat, d.end.lon]],
    points: d.strewn_points[scenario],
    line: d.fall_line,
    finds: d.finds.map((f) => ({ lat: f.lat, lon: f.lon, mass_g: Number(f.mass_g), place: f.place, explained_by: f.explained_by })),
  }, [d, scenario]);

  return (
    <main id="main">
      <ProofNav caseTitle="fireballs and meteorites" />
      <header className="page-head">
        <h1>Fireballs and meteorite falls</h1>
        <p className="tagline">
          From the cameras&apos; record of a fireball to where its meteorites landed: whether anything survived, how heavy it
          was, where it came from in the Solar System, and where to search. Tested on the Winchcombe fall of 28 February 2021,
          against the published orbit and the stones that were found.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && map && (
        <>
          <OnThisPage items={[{ id: "fb-out", label: "Result" }, { id: "fb-map", label: "Where the stones fell" }, { id: "fb-new", label: "A new fall, predicted" }, { id: "fb-ab", label: "Did anything survive" }, { id: "fb-orbit", label: "Orbit" }, { id: "fb-dark", label: "Dark flight" }]} />
          <section className="panel stack" aria-labelledby="fb-out">
            <h2 id="fb-out">Winchcombe, England</h2>
            <OutcomeCard o={d.outcome} />
            <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
              <a className="button secondary" href="/astro/winchcombe_strewn_field.geojson" download><Download size={16} aria-hidden="true" /> Strewn field (GeoJSON)</a>
              <a className="button secondary" href="/astro/winchcombe_strewn_field.kml" download><Download size={16} aria-hidden="true" /> Strewn field (KML, Google Earth)</a>
            </div>
          </section>

          <section className="panel stack" aria-labelledby="fb-new">
            <h2 id="fb-new">A new fall, predicted before anyone looked</h2>
            <p>The same blind rule, run on a fireball whose meteorites have not been reported: 2026 August 14, 07:48 UTC, over
              Douglas County, Washington (NASA&apos;s camera solution; Doppler weather radar suggests small stones reached the
              ground). Where each mass of stone should lie was registered on 2 November 2026, hashed, before any find: from about
              48.08° N, 119.35° W for 5 g stones to 48.18° N, 119.38° W for 600 g ones, each region a few hundred km² because the
              camera solution&apos;s errors are not published. It is scored when finds are reported, hit or miss.</p>
            <p><a href="/predictions">See the registered prediction and its hash</a></p>
          </section>

          <section className="panel stack" aria-labelledby="fb-map">
            <h2 id="fb-map">Where the stones fell</h2>
            <div className="row" role="radiogroup" aria-label="When the stones separated" style={{ gap: 16, flexWrap: "wrap" }}>
              <label><input type="radio" checked={scenario === "at_end"} onChange={() => setScenario("at_end")} /> blind (rule fixed before comparing): released at the luminous end</label>
              <label><input type="radio" checked={scenario === "fragmentation_zone"} onChange={() => setScenario("fragmentation_zone")} />
                {" "}after review (with published fragment information): released anywhere after the last flare ({(d.last_fragmentation_m / 1000).toFixed(1)} km)</label>
            </div>
            {d.summary.blind_inside_95 != null && (
              <p className="small"><strong>Blind score:</strong> {d.summary.blind_inside_95} of {d.summary.finds} finds lie inside the 95% landing
                region of their mass bin ({d.summary.blind_inside_68} inside the 68% one), regions computed from the simulation alone. The rule,
                fixed in advance: {d.summary.blind_rule}.</p>
            )}
            <FallMap data={map} label="Map of the simulated strewn field with the recovered stones" />
            <p className="small muted">Each dot is one simulated stone: its mass, shape (sphere to flat tile), the drag curve, the
              starting state and the radiosonde winds of the night all vary within their uncertainties. Black-ringed dots are the
              stones found (Russell et al. 2024; coordinates published to three decimals): green where simulated stones of that mass
              land, red where none do.</p>
            <h3>Every stone found, scored blind and after review ({d.finds.length})</h3>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Found</th><th>Mass</th><th>Along the track</th><th>Across</th><th>Simulated range (end / zone)</th><th>Blind region</th><th>Reproduced after review</th></tr></thead>
                <tbody>
                  {d.finds.map((f) => (
                    <tr key={`${f.place}-${f.mass_g}-${f.lat}`}>
                      <td data-label="Found">{f.place} ({f.site}){f.includes?.length ? <span className="small muted"><br />with {f.includes.join("; ")}</span> : null}</td>
                      <td data-label="Mass" className="num">{f.mass_g} g</td>
                      <td data-label="Along" className="num">{(f.along_m / 1000).toFixed(1)} km</td>
                      <td data-label="Across" className="num">{f.at_end.across_offset_m.toFixed(0)} m</td>
                      <td data-label="Simulated" className="num">{(f.at_end.along_range_m[0] / 1000).toFixed(1)} to {(f.at_end.along_range_m[1] / 1000).toFixed(1)} /
                        {" "}{(f.fragmentation_zone.along_range_m[0] / 1000).toFixed(1)} to {(f.fragmentation_zone.along_range_m[1] / 1000).toFixed(1)} km</td>
                      <td data-label="Blind region">{f.blind ? (f.blind.inside_68 ? "inside 68%" : f.blind.inside_95 ? "inside 95%" : "outside 95%") : "n/a"}</td>
                      <td data-label="Reproduced after review">{f.explained_by === "at_end" ? "yes" : f.explained_by ? "yes, if released after the flare" : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel stack" aria-labelledby="fb-ab">
            <h2 id="fb-ab">Did anything survive? The alpha-beta test</h2>
            <p className="small">Two numbers from height and speed alone: the ballistic coefficient α (drag against mass) and the
              mass-loss parameter β.</p>
            <div className="law-cell"><Tex tex={String.raw`\frac{h}{h_0} = \ln\alpha + \beta - \ln\frac{\overline{\mathrm{Ei}}(\beta) - \overline{\mathrm{Ei}}(\beta v^2)}{2}`} /></div>
            <p className="small muted">h is the height, h₀ = 7160 m the atmosphere&apos;s scale height, v = V/V₀ the speed as a fraction of the
              entry speed, Ei the exponential integral.</p>
            <div className="ab-row">
              <SpeedChart points={d.alpha_beta.points} curve={d.alpha_beta.curve} />
              <AlphaBetaDiagram events={[{ name: "Winchcombe", t: d.alpha_beta.triage }, { name: d.dfn_example.event, t: d.dfn_example.triage }]} />
            </div>
            <p className="small">Winchcombe: α = {d.alpha_beta.alpha.toFixed(1)} ± {d.alpha_beta.alpha_sigma.toFixed(1)}, β = {d.alpha_beta.beta.toFixed(2)} ±
              {" "}{d.alpha_beta.beta_sigma.toFixed(2)}: <strong>{d.alpha_beta.triage.words}</strong> (it did: 602 g were found), from the whole
              flight ({d.alpha_beta.n} speeds, typical miss {d.alpha_beta.train_rms_kms.toFixed(2)} km/s). The forecast test shows why the whole flight
              is needed: fitted on the upper 75% alone, where the stone has barely slowed, the law misses the braking below by
              {" "}{d.alpha_beta.heldout_rms_kms.toFixed(1)} km/s. Alpha-beta measures deceleration, so it cannot forecast deceleration it has not seen.
              {" "}{d.dfn_example.event} (Desert Fireball Network example): {d.dfn_example.triage.words}.</p>
            <p className="small muted">Entry mass from α (drag factor 0.8, sphere, 2090 kg/m³): {d.alpha_beta.entry_mass_kg.toFixed(0)} kg; from
              the single-body drag and ablation fit: {d.single_body.entry_mass_kg.toFixed(1)} kg; published 13 ± 3 kg. Masses from
              deceleration depend on the assumed shape and density (a factor of two in either is common), so the triage, not the
              mass, is the result here.{d.single_body.fragmentation ? ` A fragmentation was found at ${(d.single_body.fragmentation.height_m / 1000).toFixed(1)} km.`
                : " No single fragmentation beat the single body by a decisive margin in the speeds alone."}</p>
            <AgentOnCase file="agent_fireball.json" what="the fireball's distance along its track against time" />
          </section>

          <section className="panel stack" aria-labelledby="fb-orbit">
            <h2 id="fb-orbit">Where it came from</h2>
            <p className="small">The orbit before it met the Earth, from the radiant and the entry speed: the Earth&apos;s pull is
              undone exactly (the geocentric hyperbola followed back to its asymptote), then the Earth&apos;s own motion is added.</p>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Element</th><th>PhysicsLens</th><th>Published (McMullan et al. 2023)</th><th>Difference</th></tr></thead>
                <tbody>
                  {ORBIT_ROWS.map(([k, name, dp]) => (
                    <tr key={k}>
                      <td data-label="Element">{name}</td>
                      <td data-label="PhysicsLens" className="num">{d.orbit.ours[k].toFixed(dp)}</td>
                      <td data-label="Published" className="num">{d.orbit.published[k][0]} ± {d.orbit.published[k][1]}</td>
                      <td data-label="Difference" className="num">{d.orbit.pulls[k].toFixed(2)} σ</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted">Aphelion {d.orbit.ours.Q_au.toFixed(2)} AU, in the asteroid belt; Tisserand parameter
              {" "}{d.orbit.ours.tisserand_jupiter.toFixed(2)} (above 3: an asteroid&apos;s orbit, not a comet&apos;s).</p>
          </section>

          <section className="panel stack" aria-labelledby="fb-dark">
            <h2 id="fb-dark">Dark flight, against the published model</h2>
            <p className="small">Below the luminous end a stone only falls through the wind: drag with a coefficient that changes with
              the Mach number (Carter et al. 2011, sphere to cube), the air density of that night (NRLMSIS 2.1), the winds of the
              nearest radiosonde.</p>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Stone</th><th>Shape</th><th>Impact speed: here / published</th><th>Minutes falling: here / published</th></tr></thead>
                <tbody>
                  {d.dark_flight_vs_paper.map((r) => (
                    <tr key={r.stone}>
                      <td data-label="Stone">{r.stone}</td>
                      <td data-label="Shape" className="num">{r.shape[0] === r.shape[1] ? r.shape[0] : `${r.shape[0]} to ${r.shape[1]}`}</td>
                      <td data-label="Impact" className="num">{r.ours_impact_ms.toFixed(0)} / {r.impact_ms[0]} to {r.impact_ms[1]} m/s</td>
                      <td data-label="Minutes" className="num">{r.ours_minutes.toFixed(1)} / about {r.minutes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <p className="small muted">What it can&apos;t do: it starts from a triangulated trajectory (several cameras and their
            calibration), not from one video; masses from deceleration carry a factor-of-two uncertainty from shape and density;
            the winds come from the nearest radiosonde (about 120 km away), not a weather model of the night; and stones that
            separate in a late fragmentation or get a sideways kick (as Winchcombe&apos;s did) land beyond what a single release point
            predicts, which is why both release scenarios are shown. Sources: data/astro/fireball/SOURCES.md.</p>
        </>
      )}
    </main>
  );
}
