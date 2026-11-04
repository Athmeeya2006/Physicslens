"use client";

import { useEffect, useState } from "react";

// Does a calibration go stale? (scripts/robot_drift.py, data/robot/drift.json): one owner's SO-100 recorded a few days
// apart in December 2024 and again in June 2025; each recording's laws replayed on the others' held-out episodes.
// Shown only when the stored result exists; every number, and which way the verdict reads, comes from it.

interface Pair { calibration_from: string; used_on: string; days_apart: number; own_rms_deg: number; borrowed_rms_deg: number; extra_deg: number; extra_ci95: [number, number] }
interface Drift {
  recordings: { id: string; dataset: string; recorded: string; licence: string; episodes: number; own_rms_deg: number }[];
  pairs: Pair[];
  summary: { days_apart_extra_deg: number; days_apart_ci95: [number, number]; six_months_extra_deg: number;
    six_months_ci95: [number, number]; other_owners_median_extra_deg: number; drift_seen: boolean };
  limits: string[];
}

const name = (id: string) => id.split("__")[1].replace("so100_", "");
const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;

export function RobotDrift() {
  const [d, setD] = useState<Drift | null>(null);
  useEffect(() => {
    fetch("/astro/robot_drift.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const s = d.summary;
  const dated = Object.fromEntries(d.recordings.map((r) => [r.id, r.recorded]));
  const season = (id: string) => dated[id].slice(0, 4);
  const near = d.pairs.filter((p) => p.days_apart <= 10);
  const far = d.pairs.filter((p) => p.days_apart > 100);
  const late = near.filter((p) => season(p.used_on) === "2025");
  const early = near.filter((p) => season(p.used_on) === "2024");
  const own = (y: string) => d.recordings.filter((r) => r.recorded.startsWith(y)).map((r) => r.own_rms_deg);
  const farMin = Math.min(...far.map((p) => p.extra_deg)), farMax = Math.max(...far.map((p) => p.extra_deg));
  const allFarWorse = far.every((p) => p.extra_ci95[0] > 0);
  return (
    <section className="panel stack" aria-labelledby="rb-drift">
      <h2 id="rb-drift">Same arm, days and months apart: it repeats within {mean(late.map((p) => p.extra_deg)).toFixed(2)}° over three days{allFarWorse ? ", and its six-month-old calibration no longer fits" : ""}</h2>
      <p>One owner&apos;s arm, recorded twice a few days apart in December 2024 and twice again in June 2025, each recording calibrated
        on its own and replayed on the others&apos; held-out episodes. Three days apart in June, on similar tasks, the arm repeats: the
        other recording&apos;s calibration costs only <strong className="num">{mean(late.map((p) => p.extra_deg)).toFixed(2)}°</strong> extra, so a
        change of about a tenth of a degree is visible on a stable setup. That is the check a maintenance schedule needs, from logs
        the arm already writes.</p>
      <p>Borrowed across the six months, a calibration costs {farMin.toFixed(2)}° to {farMax.toFixed(2)}° extra{allFarWorse ? `, worse in all ${far.length} pairs` : ""}:
        the arm&apos;s dynamics changed between the seasons. These logs cannot say why. December&apos;s own two recordings, on different
        tasks, already disagree by {mean(early.map((p) => p.extra_deg)).toFixed(2)}°, and they fit worse even with their own calibration
        ({Math.min(...own("2024")).toFixed(2)}° to {Math.max(...own("2024")).toFixed(2)}° against {Math.min(...own("2025")).toFixed(2)}° to{" "}
        {Math.max(...own("2025")).toFixed(2)}° in June), so a change of setup between the seasons explains it as well as wear does.{" "}
        {s.drift_seen
          ? "By the rule fixed before the run, the six-month penalty exceeds the few-day one beyond both intervals: drift is claimed."
          : `By the rule fixed before the run (the six-month penalty, ${s.six_months_extra_deg.toFixed(2)}°, must exceed the few-day one, ${s.days_apart_extra_deg.toFixed(2)}°, beyond both intervals), no drift is claimed.`}</p>
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
