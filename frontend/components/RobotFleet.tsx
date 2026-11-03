"use client";

import { useEffect, useState } from "react";

// Does one calibration fit every SO-100? (scripts/robot_fleet.py, data/robot/fleet.json): each arm's identified servo
// laws replayed on every arm's held-out episodes, the constant command offset re-centred per arm. Shown only when the
// stored comparison exists; every number is read from it.

interface Cell { rms_deg: number; ci95: [number, number]; minus_own_deg: number; minus_own_ci95: [number, number] }
interface Fleet {
  arms: Record<string, { dataset: string; licence: string; episodes: number; test: number[]; params: Record<string, Record<string, { value: number }>> }>;
  matrix: Record<string, Record<string, Cell>>;
  summary: { arms: number; pairs: number; own_mean_deg: number; borrowed_mean_deg: number; ratio: number; pairs_significantly_worse: number };
  method: string;
}

const owner = (id: string) => id.split("__")[0];

export function RobotFleet() {
  const [f, setF] = useState<Fleet | null>(null);
  useEffect(() => {
    fetch("/astro/robot_fleet.json").then((r) => (r.ok ? r.json() : null)).then(setF).catch(() => setF(null));
  }, []);
  if (!f) return null;
  const ids = Object.keys(f.matrix);
  const s = f.summary;
  return (
    <section className="panel stack" aria-labelledby="rb-fleet">
      <h2 id="rb-fleet">{s.ratio > 1.1 && s.pairs_significantly_worse > s.pairs / 2
        ? "One calibration does not fit every arm" : "Does one calibration fit every arm?"}</h2>
      <p>The same identification on {s.arms} SO-100 arms recorded by different owners: each arm&apos;s laws were replayed on
        every arm&apos;s held-out episodes. Its own calibration tracks to <strong className="num">{s.own_mean_deg.toFixed(2)}°</strong> on
        average; a calibration borrowed from another arm to <strong className="num">{s.borrowed_mean_deg.toFixed(2)}°</strong> ({s.ratio.toFixed(1)} times
        the error), worse in {s.pairs_significantly_worse} of {s.pairs} pairs at 95%.</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Arm (episodes held out)</th>{ids.map((c) => <th key={c} className="num">laws of {owner(c)}</th>)}</tr></thead>
          <tbody>
            {ids.map((t) => (
              <tr key={t}>
                <td>{owner(t)} ({f.arms[t].test.length})</td>
                {ids.map((c) => (
                  <td key={c} className="num">{c === t ? <strong>{f.matrix[t][c].rms_deg.toFixed(2)}°</strong> : `${f.matrix[t][c].rms_deg.toFixed(2)}°`}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Held-out RMS over the five arm joints, from replaying the servo laws themselves (a stricter full
        MuJoCo replay of the first arm gives 0.38° above); bold is each arm&apos;s own calibration. Each owner recorded a different
        task with their own setup, so a borrowed calibration carries both the arm and the setup. Data: {ids.map((t) => `${f.arms[t].dataset} (${f.arms[t].licence})`).join("; ")}.</p>
    </section>
  );
}
