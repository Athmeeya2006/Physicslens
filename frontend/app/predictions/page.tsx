"use client";

import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { ProofNav } from "@/components/ProofNav";
import { StatusChip } from "@/components/StatusChip";

// Every pre-registered prediction (predictions/, scripts/build_predictions.py): made and hashed before the answer
// existed, scored later in outcomes/. Misses stay visible; a file changed after registration is flagged, not hidden.

interface Item {
  label: string; value?: number; unit?: string; ci68?: [number, number]; ci95?: [number, number]; utc_approx?: string;
  sigma_days?: number; shift_min?: number; mass_kg?: number; median_lat_lon?: [number, number]; hdr95_area_km2?: number;
}

// One predicted quantity in plain words: a transit as a UTC time with its 95% window and how far the published
// timetable is off; a meteorite mass bin as where its stones should lie and how big the 95% region is.
function describe(it: Item): string {
  if (it.utc_approx && it.sigma_days != null) {
    const w = (1.96 * it.sigma_days * 1440).toFixed(1);
    const shift = it.shift_min != null ? `; the published timetable says ${Math.abs(it.shift_min).toFixed(0)} min ${it.shift_min < 0 ? "later" : "earlier"}` : "";
    return `${it.label}: ${it.utc_approx.replace("T", " ").replace("Z", " UTC")} ± ${w} min (95%)${shift}`;
  }
  if (it.median_lat_lon) {
    const [la, lo] = it.median_lat_lon;
    return `${it.label}: centred on ${la.toFixed(3)}° N, ${Math.abs(lo).toFixed(3)}° ${lo < 0 ? "W" : "E"}` +
      (it.hdr95_area_km2 != null ? `, 95% region ${it.hdr95_area_km2.toFixed(0)} km²` : "");
  }
  return `${it.label}: ${it.value ?? ""}${it.unit ? ` ${it.unit}` : ""}${it.ci95 ? ` (95%: ${it.ci95[0]} to ${it.ci95[1]})` : ""}`;
}
interface Outcome { scored_utc: string; summary: string; hits?: number; misses?: number }
interface Pred {
  created_utc: string; case: string; id: string; score_after_utc: string; sha256: string; title?: string; question?: string;
  prediction?: { items?: Item[] }; scoring?: { rule?: string }; intact: boolean; outcome: Outcome | null;
}
interface Data { predictions: Pred[]; open: number; scored: number; hits: number; misses: number; integrity_problems: string[] }

const REPO = "https://github.com/Athmeeya2006/Physicslens/blob/main";

function day(s: string): string {
  return s.slice(0, 10);
}

export default function PredictionsPage() {
  const [d, setD] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/predictions.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The predictions could not be loaded (${(e as Error).message}).`));
  }, []);
  return (
    <main id="main">
      <ProofNav />
      <header className="page-head">
        <h1>Predictions made before the answer</h1>
        <p className="tagline">
          A forecast made after the answer is known can always be bent to fit it. Each one here was written to a file, hashed
          and committed before reality could answer; later it is scored, hit or miss, and misses stay on this page.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <ul className="stat-strip count-strip" aria-label="Running score">
            <li><span className="stat-number">{d.open}</span><span className="stat-label">still open</span></li>
            <li><span className="stat-number">{d.scored}</span><span className="stat-label">scored</span></li>
            <li><span className="stat-number">{d.hits}</span><span className="stat-label">hits</span></li>
            <li><span className="stat-number">{d.misses}</span><span className="stat-label">misses</span></li>
          </ul>
          {d.integrity_problems.length > 0 && <p className="error" role="alert">Changed after registration: {d.integrity_problems.join("; ")}</p>}
          <section className="stack" style={{ gap: 12 }}>
            <h2>How each prediction is locked in</h2>
            <ol>
              <li><strong>Written before the answer exists.</strong> Each prediction is a file with its inputs and their
                sources, the forecast with 68% and 95% intervals, the rule that will score it, the date after which it can
                be scored, and the exact code version.</li>
              <li><strong>Hashed and timestamped.</strong> Its SHA-256 goes into <code>predictions/INDEX.md</code> the moment
                it is written; the public repository&apos;s history is the timestamp.</li>
              <li><strong>Never edited.</strong> A corrected prediction is a new one, and the old one stays listed. This page
                recomputes every hash and warns if a file changed.</li>
              <li><strong>Scored by reality.</strong> After its date, what actually happened is written next to it, hit or
                miss, and misses stay here.</li>
            </ol>
            <p className="small">Registered so far: transit times of eight exoplanets whose published timetables have drifted
              by up to hours (refreshed from newer TESS data and checked on a held-out season first), and where meteorites
              from a new fireball should be found. Each row below says when it can first be scored.</p>
          </section>
          {d.predictions.length === 0 && <p className="muted">No prediction is registered yet: the first ones appear here
            as soon as their files are written.</p>}
          <ul className="case-rows">
            {d.predictions.map((p, i) => (
              <li key={`${p.case}/${p.id}`}>
                <details className="case-row" open={i === 0}>
                  <summary>
                    <span className="case-row-status">
                      <StatusChip status={p.outcome ? (p.outcome.hits ? "hit" : "miss") : "predicted"} label={p.outcome ? (p.outcome.hits ? "Hit" : "Miss") : "Open"} />
                      <span className="small">{p.outcome ? `scored ${day(p.outcome.scored_utc)}` : `scored after ${day(p.score_after_utc)}`}</span>
                    </span>
                    <span className="case-row-main">
                      <strong className="case-row-title" id={`pred-${p.id}`}>{p.title ?? p.id}</strong>
                      {p.question && <span className="small">{p.question}</span>}
                      {p.outcome && <span className="case-row-number">{p.outcome.summary}</span>}
                    </span>
                    <ChevronDown size={20} className="case-row-chev" aria-hidden="true" />
                  </summary>
                  <div className="case-row-body">
                    <p className="small">Predicted on {day(p.created_utc)}{p.outcome ? `, scored on ${day(p.outcome.scored_utc)}` : `, can be scored after ${day(p.score_after_utc)}`}.</p>
                    {p.prediction?.items && (
                      <ul className="small">
                        {p.prediction.items.slice(0, 8).map((it) => (
                          <li key={it.label}>{describe(it)}</li>
                        ))}
                        {p.prediction.items.length > 8 && <li>and {p.prediction.items.length - 8} more in the file</li>}
                      </ul>
                    )}
                    <dl className="facts">
                      {p.scoring?.rule && <><dt>Scored by</dt><dd>{p.scoring.rule}</dd></>}
                      <dt>File</dt><dd>
                        <a href={`${REPO}/predictions/${p.case}/${p.id}/prediction.json`}>prediction.json</a>, SHA-256 <code>{p.sha256.slice(0, 16)}…</code>
                        {p.intact ? " (matches the registered hash)" : " (does NOT match the registered hash)"}
                      </dd>
                    </dl>
                  </div>
                </details>
              </li>
            ))}
          </ul>
        </>
      )}
    </main>
  );
}
