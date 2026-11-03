"use client";

import { useEffect, useState } from "react";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { ProofNav } from "@/components/ProofNav";
import { Term } from "@/components/Term";

// Umbriel's outline from the 2020 stellar occultation (scripts/occultation_demo.py): every observer's chord in the
// plane perpendicular to the star, the fitted circle, and the observers who saw nothing. Precomputed, served static.

interface ChordRow { observer: string; positive: boolean; points: [number, number][]; track: [number, number][] | null; sigma_km: number[] | null }
interface Fit { params: Record<string, number>; sigma: Record<string, number>; chi2: number; dof: number; limb_sigma_km: number;
  misses_crossed: number; equivalent_radius?: number }
interface Occ {
  event: string; shadow_speed_km_s: number; chords: ChordRow[]; fits: { circle: Fit; ellipse: Fit };
  published: { radius: [number, number]; a_prime: [number, number]; R_eq_prime: [number, number]; limb_sigma_km: number; source: string };
  edge_trials: { exposure_s: number; rms_error_s: number; within_1_sigma: number }[];
  outcome: Outcome;
}

function ChordPlot({ d }: { d: Occ }) {
  const c = d.fits.circle.params;
  const S = 360, half = 1.25 * c.radius;
  const X = (f: number) => S / 2 - ((f - c.fc) / half) * (S / 2);  // east (f) to the left, as on the sky
  const Y = (g: number) => S / 2 - ((g - c.gc) / half) * (S / 2);
  const e = d.fits.ellipse.params;
  return (
    <figure className="ab-chart" style={{ maxWidth: 520 }}>
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Chords of the observers across Umbriel's shadow, the fitted circle and ellipse, and the misses">
        <rect x={0} y={0} width={S} height={S} className="ab-frame" />
        <circle cx={X(c.fc)} cy={Y(c.gc)} r={(c.radius / half) * (S / 2)} fill="var(--model)" opacity={0.12} stroke="var(--model)" strokeWidth={2} />
        <ellipse cx={X(e.fc)} cy={Y(e.gc)} rx={(e.a / half) * (S / 2)} ry={(e.b / half) * (S / 2)} fill="none" stroke="var(--tie)" strokeDasharray="5 4"
                 transform={`rotate(${(e.phi * 180) / Math.PI} ${X(e.fc)} ${Y(e.gc)})`} />
        {d.chords.filter((ch) => !ch.positive && ch.track).map((ch) => (
          <polyline key={ch.observer} points={ch.track!.map(([f, g]) => `${X(f).toFixed(1)},${Y(g).toFixed(1)}`).join(" ")} fill="none"
                    stroke="var(--bad)" strokeWidth={1.2} strokeDasharray="2 3"><title>{`${ch.observer}: saw nothing`}</title></polyline>
        ))}
        {d.chords.filter((ch) => ch.positive).map((ch) => (
          <g key={ch.observer}>
            <line x1={X(ch.points[0][0])} y1={Y(ch.points[0][1])} x2={X(ch.points[1][0])} y2={Y(ch.points[1][1])} stroke="var(--ink)" strokeWidth={1.4}>
              <title>{`${ch.observer}: ±${ch.sigma_km![0].toFixed(1)} / ±${ch.sigma_km![1].toFixed(1)} km at the edges`}</title>
            </line>
            {ch.points.map(([f, g], k) => <circle key={k} cx={X(f)} cy={Y(g)} r={1.8} className="oc-tra" />)}
          </g>
        ))}
        <text x={8} y={16} className="tick">north up, east left; {Math.round(2 * half)} km across</text>
      </svg>
      <figcaption className="small">
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--ink)" strokeWidth="1.4" /></svg>an observer who saw the star vanish</span>
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--bad)" strokeDasharray="2 3" /></svg>an observer who saw nothing</span>
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--model)" strokeWidth="2" /></svg>fitted circle</span>
        <span className="key"><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="var(--tie)" strokeDasharray="5 4" /></svg>fitted ellipse</span>
      </figcaption>
    </figure>
  );
}

export default function OccultationPage() {
  const [d, setD] = useState<Occ | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/astro/occultation.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The occultation results could not be loaded (${(e as Error).message}).`));
  }, []);
  const c = d?.fits.circle;
  const e = d?.fits.ellipse;

  return (
    <main id="main">
      <ProofNav caseTitle="stellar occultations" />
      <header className="page-head">
        <h1>Stellar occultations</h1>
        <p className="tagline">
          When an asteroid or a moon passes in front of a star, everyone in its shadow sees the star vanish for a few seconds.
          Each observer&apos;s timing is one line across the body&apos;s outline; together they measure its size and shape
          directly. Tested on Uranus&apos;s moon Umbriel, from the published timings of 2020 September 21.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && c && e && (
        <>
          <section className="panel stack" aria-labelledby="occ-out">
            <h2 id="occ-out">Umbriel&apos;s size from one star blinking out</h2>
            <OutcomeCard o={d.outcome} />
            <ChordPlot d={d} />
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Fit</th><th>PhysicsLens</th><th>Published</th></tr></thead>
                <tbody>
                  <tr><td data-label="Fit">Circle radius</td><td data-label="PhysicsLens" className="num">{c.params.radius.toFixed(1)} ± {c.sigma.radius.toFixed(1)} km</td>
                    <td data-label="Published" className="num">{d.published.radius[0]} ± {d.published.radius[1]} km</td></tr>
                  <tr><td data-label="Fit">Ellipse semi-major axis</td><td data-label="PhysicsLens" className="num">{e.params.a.toFixed(1)} ± {e.sigma.a.toFixed(1)} km</td>
                    <td data-label="Published" className="num">{d.published.a_prime[0]} ± {d.published.a_prime[1]} km</td></tr>
                  <tr><td data-label="Fit">Ellipse equivalent radius</td><td data-label="PhysicsLens" className="num">{e.equivalent_radius?.toFixed(1)} km</td>
                    <td data-label="Published" className="num">{d.published.R_eq_prime[0]} ± {d.published.R_eq_prime[1]} km</td></tr>
                  <tr><td data-label="Fit">Limb roughness added to each edge</td><td data-label="PhysicsLens" className="num">{c.limb_sigma_km.toFixed(1)} km</td>
                    <td data-label="Published" className="num">{d.published.limb_sigma_km} km</td></tr>
                </tbody>
              </table>
            </div>
            <p className="small muted">{d.chords.filter((x) => x.positive).length} chords and {d.chords.filter((x) => !x.positive).length} misses; the shadow crossed the
              plane at {d.shadow_speed_km_s.toFixed(1)} km/s, so a tenth of a second of timing is {(d.shadow_speed_km_s / 10).toFixed(1)} km on the outline. The
              ellipse&apos;s orientation is left free here (the paper fixed it from the moon&apos;s expected pose), which is why its axes differ more.
              Source: {d.published.source}.</p>
          </section>
          <section className="panel stack" aria-labelledby="occ-edge">
            <h2 id="occ-edge">Timing an edge inside a frame</h2>
            <p className="small">A camera frame integrates light over its exposure, so the star can vanish partway through a frame. The fit models that, and its
              errors are calibrated: on synthetic light curves with known edges, the measured time lands within one stated sigma about two thirds of the time.</p>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Exposure</th><th>Typical timing error</th><th>Within one <Term k="sigma">sigma</Term></th></tr></thead>
                <tbody>{d.edge_trials.map((t) => (
                  <tr key={t.exposure_s}><td data-label="Exposure" className="num">{t.exposure_s} s</td><td data-label="Error" className="num">{(t.rms_error_s * 1000).toFixed(0)} ms</td>
                    <td data-label="Within" className="num">{(t.within_1_sigma * 100).toFixed(0)}% (68% expected)</td></tr>))}</tbody>
              </table>
            </div>
          </section>
          <p className="small muted">Scope: the chords here are the published timings (no raw videos were reprocessed); diffraction at the
            edges is not modelled (it matters only for very fast cameras or very distant bodies); and a few chords can fit many
            shapes, which is why misses near the edge are kept as constraints rather than thrown away.</p>
        </>
      )}
    </main>
  );
}
