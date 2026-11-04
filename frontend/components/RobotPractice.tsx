"use client";

import { useEffect, useState } from "react";

// What the corrected SO-100 simulator changes in practice (scripts/robot_impact.py, data/robot/impact.json): where the
// gripper ends up, in millimetres, and a compensator built from the same laws. Shown only when the stored result
// exists; every number is read from it.

interface Sum { mean: number; ci95: [number, number]; median: number; p90: number; n: number }
interface Grip { label: string; rms_mm: Sum; at_pick_mm: Sum; at_place_mm: Sum }
interface Impact {
  test_episodes: number[];
  gripper: Record<string, Grip>;
  gripper_ratio_default_over_identified: Sum;
  compensation: Record<"real_data" | "simulation", { what: string; rows: Record<string, Sum>; labels: Record<string, string> }>;
  limits: string[];
}

const mm = (x: number) => `${x.toFixed(1)} mm`;

export function RobotPractice() {
  const [d, setD] = useState<Impact | null>(null);
  useEffect(() => {
    fetch("/astro/robot_impact.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const g = d.gripper;
  const real = d.compensation.real_data;
  const sim = d.compensation.simulation;
  const ratio = real.rows.delay_and_offset.mean / real.rows.compensator.mean;
  return (
    <>
      <section className="panel stack" aria-labelledby="rb-mm">
        <h2 id="rb-mm">Where the gripper ends up: {mm(g.default.rms_mm.mean)} off with the public model, {mm(g.identified.rms_mm.mean)} corrected</h2>
        <p>Degrees at the joints become millimetres at the hand. Each held-out replay was turned into the position of the
          gripper&apos;s fixed jaw tip and compared with where the real arm&apos;s own joint readings put it. Over the motion the
          corrected simulator keeps the hand <strong className="num">{d.gripper_ratio_default_over_identified.mean.toFixed(1)} times</strong> closer
          to the real one; when the arm opens to set an object down, it puts the hand within {mm(g.identified.at_place_mm.median)} (median)
          against {mm(g.default.at_place_mm.median)}. At the pick itself the arm has slowed almost to a stop, so every model is close there.</p>
        <div className="table-wrap">
          <table className="cards">
            <thead><tr><th>Simulator</th><th className="num">Hand error over the motion (RMS)</th><th className="num">At the pick (median)</th><th className="num">At the place (median)</th></tr></thead>
            <tbody>
              {["default", "baseline_delay_offset", "identified"].map((k) => (
                <tr key={k}>
                  <td data-label="Simulator">{k === "identified" ? <strong>{g[k].label}</strong> : g[k].label}</td>
                  <td data-label="Over the motion" className="num">{mm(g[k].rms_mm.mean)} <span className="muted small">({g[k].rms_mm.ci95[0].toFixed(1)} to {g[k].rms_mm.ci95[1].toFixed(1)})</span></td>
                  <td data-label="At the pick" className="num">{mm(g[k].at_pick_mm.median)}</td>
                  <td data-label="At the place" className="num">{mm(g[k].at_place_mm.median)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted">{d.test_episodes.length} held-out episodes, the same replays as above. Tuning the public model with a
          delay and offsets lowers the joint error, but its gains are still guessed, so the hand ends up no closer at the pick or the place.</p>
      </section>
      <section className="panel stack" aria-labelledby="rb-comp">
        <h2 id="rb-comp">The same laws, run backwards: a compensator</h2>
        <p>Inverting each joint&apos;s law gives the command that makes the servo follow a planned path: the path a delay ahead, plus
          the lag it needs, plus a step over the deadband. Checked on real data first: run backwards on the held-out episodes, it
          recovers the command that was really sent within <strong className="num">{real.rows.compensator.mean.toFixed(2)}°</strong>, against{" "}
          {real.rows.delay_and_offset.mean.toFixed(2)}° for a fitted delay and offsets ({ratio.toFixed(1)} times closer) and{" "}
          {real.rows.send_the_path.mean.toFixed(2)}° for assuming the arm does what it is told.</p>
        <p>In simulation, with the corrected model standing in for the arm, sending the compensated command instead of the plan
          itself cuts the tracking error from {sim.rows.send_the_path.mean.toFixed(2)}° to <strong className="num">{sim.rows.compensator.mean.toFixed(2)}°</strong>.</p>
      </section>
    </>
  );
}
