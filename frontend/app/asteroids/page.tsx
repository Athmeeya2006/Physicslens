"use client";

import { Download } from "lucide-react";
import { useEffect, useState } from "react";
import { OutcomeCard } from "@/components/OutcomeCard";
import type { Outcome } from "@/lib/api";
import { ProofNav } from "@/components/ProofNav";

// Did my observation matter? Asteroid 2024 YR4 from its first nights (scripts/asteroid_demo.py): the orbit's
// uncertainty a month ahead before and after one more night, checked against later observations. Static.

interface Side { rms_arcsec: number; rejected: number; uncertainty_ahead: { major_arcsec: number; minor_arcsec: number };
  forecast_check: { n: number; median_miss_arcsec: number; max_miss_arcsec: number; predicted_1sigma_major_arcsec: number; within_3_sigma: number } }
interface Ast {
  object: string; first_nights: string[]; your_night: [string, string]; n_first: number; n_yours: number; ahead_days: number; ahead_utc: string;
  before: Side; after: Side; your_residuals_arcsec: { ra: number[]; dec: number[]; rms: number }; check_window: [string, string];
  outcome: Outcome;
}

function Ellipses({ d }: { d: Ast }) {
  const S = 300, max = d.before.uncertainty_ahead.major_arcsec * 1.15;
  const sc = (v: number) => (v / max) * (S / 2);
  return (
    <figure className="ab-chart" style={{ maxWidth: 360 }}>
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Where the asteroid could be a month ahead, before and after one more night">
        <rect x={0} y={0} width={S} height={S} className="ab-frame" />
        <ellipse cx={S / 2} cy={S / 2} rx={sc(d.before.uncertainty_ahead.major_arcsec)} ry={sc(d.before.uncertainty_ahead.minor_arcsec)}
                 fill="var(--tie)" opacity={0.25} stroke="var(--tie)" />
        <ellipse cx={S / 2} cy={S / 2} rx={sc(d.after.uncertainty_ahead.major_arcsec)} ry={sc(d.after.uncertainty_ahead.minor_arcsec)}
                 fill="var(--model)" opacity={0.5} stroke="var(--model)" />
        <text x={8} y={16} className="tick">{(2 * max).toFixed(1)} arcsec across (1-sigma ellipses)</text>
      </svg>
      <figcaption className="small">
        <span className="key"><span className="dot" style={{ background: "var(--tie)", width: 10, height: 10, borderRadius: "50%", display: "inline-block" }} />first {d.first_nights.length} nights</span>
        <span className="key"><span className="dot" style={{ background: "var(--model)", width: 10, height: 10, borderRadius: "50%", display: "inline-block" }} />plus one more night</span>
      </figcaption>
    </figure>
  );
}

// Residuals against JPL (scripts/asteroid_jpl_check.py): JPL Horizons' position from each site at each time.
interface Jpl {
  your_night: { station: string; date: string; n: number; residual_rms_arcsec: number; stated_sigma_arcsec: number };
  forecast_vs_jpl: { n: number; median_arcsec: number; max_arcsec: number; window: [string, string] };
  observations_vs_jpl: { median_arcsec: number };
}

export default function AsteroidsPage() {
  const [d, setD] = useState<Ast | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jpl, setJpl] = useState<Jpl | null>(null);
  useEffect(() => {
    fetch("/astro/asteroids_jpl.json").then((r) => (r.ok ? r.json() : null)).then(setJpl).catch(() => setJpl(null));
    fetch("/astro/asteroids.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The asteroid results could not be loaded (${(e as Error).message}).`));
  }, []);

  return (
    <main id="main">
      <ProofNav caseTitle="asteroid follow-up" />
      <header className="page-head">
        <h1>Asteroid follow-up</h1>
        <p className="tagline">
          A newly found asteroid is lost unless someone observes it again soon. How much does one more night help? Tested on
          2024 YR4, the asteroid whose impact odds made headlines in early 2025, from the observations the Minor Planet Center
          holds.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <section className="panel stack" aria-labelledby="ast-out">
            <h2 id="ast-out">Did my observation matter?</h2>
            <OutcomeCard o={d.outcome} />
            <Ellipses d={d} />
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th></th><th>First {d.first_nights.length} nights ({d.n_first} positions)</th><th>Plus one night ({d.n_yours} positions)</th></tr></thead>
                <tbody>
                  <tr><td data-label="">Uncertainty on {d.ahead_utc} (1 sigma)</td>
                    <td data-label="Before" className="num">{d.before.uncertainty_ahead.major_arcsec.toFixed(2)}″</td>
                    <td data-label="After" className="num">{d.after.uncertainty_ahead.major_arcsec.toFixed(2)}″</td></tr>
                  <tr><td data-label="">Miss on the observations of {d.check_window[0]} to {d.check_window[1]} (median, max)</td>
                    <td data-label="Before" className="num">{d.before.forecast_check.median_miss_arcsec.toFixed(2)}″, {d.before.forecast_check.max_miss_arcsec.toFixed(2)}″</td>
                    <td data-label="After" className="num">{d.after.forecast_check.median_miss_arcsec.toFixed(2)}″, {d.after.forecast_check.max_miss_arcsec.toFixed(2)}″</td></tr>
                  <tr><td data-label="">Later observations within 3 stated sigmas</td>
                    <td data-label="Before" className="num">{(d.before.forecast_check.within_3_sigma * 100).toFixed(0)}% of {d.before.forecast_check.n}</td>
                    <td data-label="After" className="num">{(d.after.forecast_check.within_3_sigma * 100).toFixed(0)}% of {d.after.forecast_check.n}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="small muted">The extra night stands in for yours: the four positions ATLAS Chile (station {d.your_night[1]}) measured on {d.your_night[0]}. Against the
              orbit fitted without them they sit {d.your_residuals_arcsec.rms.toFixed(2)}″ off (rms), the kind of check that tells you your setup and timing are sound.
              The orbit: Newton&apos;s pull of the Sun, the planets and the Moon, light-time for every position, weighted by each observation&apos;s stated
              uncertainty, outliers beyond 3 sigma set aside. JPL&apos;s orbit is used only as the starting guess; started 30,000 km away the fit lands on the
              same orbit.</p>
          </section>
          {jpl && (
            <section className="panel stack" aria-labelledby="ast-jpl">
              <h2 id="ast-jpl">Against JPL</h2>
              <dl className="facts">
                <dt>Your setup</dt>
                <dd>The {jpl.your_night.n} positions of {jpl.your_night.date} (station {jpl.your_night.station}) sit{" "}
                  <strong className="num">{jpl.your_night.residual_rms_arcsec.toFixed(2)}″</strong> (rms) from JPL&apos;s position for that site and
                  time, inside the {jpl.your_night.stated_sigma_arcsec.toFixed(2)}″ the observatory states. Residuals near the stated
                  uncertainty mean a sound setup and clock; a large mean offset points at the timing.</dd>
                <dt>The forecast</dt>
                <dd>Over {jpl.forecast_vs_jpl.window[0]} to {jpl.forecast_vs_jpl.window[1]}, the orbit from three nights lands a median{" "}
                  <strong className="num">{jpl.forecast_vs_jpl.median_arcsec.toFixed(2)}″</strong> (at most {jpl.forecast_vs_jpl.max_arcsec.toFixed(2)}″) from
                  JPL&apos;s orbit, which uses every observation made since, at {jpl.forecast_vs_jpl.n} later positions.</dd>
                <dt>The noise floor</dt>
                <dd>The real observations of that window sit a median {jpl.observations_vs_jpl.median_arcsec.toFixed(2)}″ from JPL&apos;s orbit.</dd>
              </dl>
            </section>
          )}
          <section className="panel stack" aria-labelledby="ast-ades">
            <h2 id="ast-ades">The report</h2>
            <p className="small">Observations in ADES (the Minor Planet Center&apos;s format, version 2022), ready for your own MPC observatory code. PhysicsLens
              never submits anything: you do.</p>
            <p><a className="button secondary" href="/astro/example_ades.psv" download><Download size={16} aria-hidden="true" /> Example ADES report (PSV)</a></p>
          </section>
          <p className="small muted">Scope: it works alongside the free astrometry tools (Find_Orb, the MPC&apos;s own services) and adds a verified forecast with its
            uncertainty; the Yarkovsky drift needs years of observations, beyond one night; and this page starts from published positions (the
            frame-to-position steps are in the real sky data pipeline). Data: Minor Planet Center observations (public API), JPL Horizons (starting guess and the comparison above).</p>
        </>
      )}
    </main>
  );
}
