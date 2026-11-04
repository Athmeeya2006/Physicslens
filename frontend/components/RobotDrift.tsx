"use client";

import { useEffect, useState } from "react";

// Does a calibration go stale? (scripts/robot_drift.py, data/robot/drift.json): one owner's SO-100 recorded a few days
// apart in December 2024 and again in June 2025; each recording's laws replayed on the others' held-out episodes.
// Shown only when the stored result exists; every number, and which way the headline reads, comes from it.

interface Pair { calibration_from: string; used_on: string; days_apart: number; own_rms_deg: number; borrowed_rms_deg: number; extra_deg: number; extra_ci95: [number, number] }
interface Drift {
  recordings: { id: string; dataset: string; recorded: string; licence: string; episodes: number; own_rms_deg: number }[];
  pairs: Pair[];
  summary: { days_apart_extra_deg: number; days_apart_ci95: [number, number]; days_apart_pairs: number; six_months_extra_deg: number;
    six_months_ci95: [number, number]; six_months_pairs: number; other_owners_median_extra_deg: number; drift_seen: boolean };
  limits: string[];
}

const name = (id: string) => id.split("__")[1].replace("so100_", "");

export function RobotDrift() {
  const [d, setD] = useState<Drift | null>(null);
  useEffect(() => {
    fetch("/astro/robot_drift.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const s = d.summary;
  const dated = Object.fromEntries(d.recordings.map((r) => [r.id, r.recorded]));
  return (
    <section className="panel stack" aria-labelledby="rb-drift">
      <h2 id="rb-drift">{s.drift_seen ? "Six months later the same arm needs a new calibration" : "Six months later: does the calibration still hold?"}</h2>
      <p>One owner&apos;s arm, recorded twice a few days apart in December 2024 and twice again in June 2025. A calibration a few days
        old costs <strong className="num">{s.days_apart_extra_deg.toFixed(2)}°</strong> extra on average (95%: {s.days_apart_ci95[0].toFixed(2)} to{" "}
        {s.days_apart_ci95[1].toFixed(2)}): that is how well one calibration repeats across tasks. One six months old costs{" "}
        <strong className="num">{s.six_months_extra_deg.toFixed(2)}°</strong> ({s.six_months_ci95[0].toFixed(2)} to {s.six_months_ci95[1].toFixed(2)}).{" "}
        {s.drift_seen
          ? "The gap is larger than the repeat error, so the arm's dynamics changed between the two seasons: the check a maintenance schedule needs, from logs the arm already writes."
          : "The two are not separated at 95%, so no change of the arm's dynamics shows over the six months; the same check, run on newer logs, would show one when it comes."}{" "}
        For scale, a calibration from another owner&apos;s arm costs {s.other_owners_median_extra_deg.toFixed(2)}° (median).</p>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Calibration from</th><th>Used on</th><th className="num">Apart</th><th className="num">Extra error (95%)</th></tr></thead>
          <tbody>
            {[...d.pairs].sort((a, b) => a.days_apart - b.days_apart).map((p) => (
              <tr key={`${p.calibration_from}-${p.used_on}`}>
                <td data-label="Calibration from">{name(p.calibration_from)} ({dated[p.calibration_from]})</td>
                <td data-label="Used on">{name(p.used_on)} ({dated[p.used_on]})</td>
                <td data-label="Apart" className="num">{p.days_apart} days</td>
                <td data-label="Extra error" className="num">{p.extra_deg.toFixed(2)}° <span className="muted small">({p.extra_ci95[0].toFixed(2)} to {p.extra_ci95[1].toFixed(2)})</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Extra error: the borrowed calibration&apos;s held-out RMS over the five arm joints minus the recording&apos;s own.
        {" "}{d.limits.join("; ")}. Data: {d.recordings.map((r) => `${r.dataset} (${r.licence})`).join("; ")}.</p>
    </section>
  );
}
