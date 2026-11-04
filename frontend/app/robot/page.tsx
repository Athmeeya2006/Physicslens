"use client";

import { Download } from "lucide-react";
import { useEffect, useState } from "react";
import { OnThisPage } from "@/components/OnThisPage";
import { ProofNav } from "@/components/ProofNav";
import { RobotDrift } from "@/components/RobotDrift";
import { RobotFleet } from "@/components/RobotFleet";
import { RobotPolicy } from "@/components/RobotPolicy";
import { RobotPractice } from "@/components/RobotPractice";
import { RobotTrace, type Trace } from "@/components/RobotTrace";

// The SO-100 arm case (scripts/robot_demo.py, data/robot/result.json): per-joint servo laws identified from public
// LeRobot recordings, the corrected MuJoCo model, and the held-out comparison with the public model and fair baselines.

interface Ci { value: number; ci95: [number, number] }
interface Joint { law: string; title: string; params: Record<string, Ci & { at_bound?: boolean }> }
interface Rms { rms_deg: number; ci95: [number, number] }
interface Robot {
  status: string; status_rule: string; limits: string[];
  data: { dataset: string; licence: string; episodes: number; frames: number; fps: number; fit_episodes: number[];
    select_episodes: number[]; test_episodes: number[] };
  model: { source: string; licence: string };
  joints: Record<string, Joint>;
  heldout: Record<string, { joints: Record<string, Rms>; arm_total: Rms }>;
  heldout_comparison: Record<string, { improvement_deg: number; ci95: [number, number]; ratio: number }>;
  fair_baseline: { what: string; delay_s: number }; stronger_baseline: { what: string };
  trace?: Trace;
}

const PARAM: Record<string, string> = { tau: "time constant (s)", d: "delay (s)", b: "command offset (deg)", db: "deadband (deg)",
  wn: "natural frequency (rad/s)", zeta: "damping ratio", wmax: "rate limit (deg/s)" };
const MODELS: [string, string][] = [
  ["default", "MuJoCo Menagerie model, as published"],
  ["default_no_gravity", "Published model without gravity (shows how much the joint map matters)"],
  ["baseline_delay", "Fair baseline: published model + one fitted delay"],
  ["baseline_delay_offset", "Stronger baseline: + per-joint offsets"],
  ["identified_mjcf_only", "Identified MJCF parameters alone (no command offsets)"],
  ["identified_not_recentred", "Identified, without the training-set recentring"],
  ["identified", "Identified servo dynamics in MuJoCo (this case)"],
  ["identified_servo_laws", "The identified servo laws on their own, outside MuJoCo"],
];

export default function RobotPage() {
  const [d, setD] = useState<Robot | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/astro/robot.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setD).catch((e) => setError(`The robot results could not be loaded (${(e as Error).message}).`));
  }, []);
  const joints = d ? Object.keys(d.heldout.default.joints) : [];
  return (
    <main id="main">
      <ProofNav caseTitle="open-source robot arm" />
      <header className="page-head">
        <h1>Open-source robot arm: a simulator twice as true to the real arm as the public one</h1>
        <p className="tagline">
          The SO-100 is a widely used low-cost arm, and its public MuJoCo model has guessed joint dynamics. From public
          recordings of the real arm (the command sent and the angle reached, every frame), each joint&apos;s servo law is
          identified and checked on episodes it never saw, then put back into the MuJoCo model.
        </p>
        <p className="tagline">
          The public model is the one people simulate this arm with and test its policies against. On motion it never saw, the
          corrected model tracks the real arm about twice as closely, and it stays ahead of the fixes an engineer would try
          first: a fitted delay, then per-joint offsets on top. And every arm needs its own: on seven owners&apos; arms, a
          calibration borrowed from another arm is worse than the arm&apos;s own in every pair.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {d && (
        <>
          <OnThisPage items={[{ id: "rb-out", label: "Result" }, { id: "rb-joints", label: "Each joint's law" }, { id: "rb-trace", label: "Before and after" },
            { id: "rb-mm", label: "At the gripper" }, { id: "rb-comp", label: "Compensator" }, { id: "rb-fleet", label: "Seven arms" }, { id: "rb-policy", label: "Under an AI policy" },
            { id: "rb-drift", label: "Six months later" }, { id: "rb-cmp", label: "Against the public model" }, { id: "rb-data", label: "Data" }]} />
          <section className="panel stack" aria-labelledby="rb-out">
            <h2 id="rb-out">Result</h2>
            <p className="case-number">On {d.data.test_episodes.length} held-out episodes the corrected model tracks the real arm to{" "}
              {d.heldout.identified.arm_total.rms_deg.toFixed(2)}° RMS, against {d.heldout.default.arm_total.rms_deg.toFixed(2)}° for the public
              model: {d.heldout_comparison.default_minus_identified.ratio.toFixed(1)} times closer, and{" "}
              {d.heldout_comparison.fair_baseline_minus_identified.ratio.toFixed(1)} times closer than the public model with a fitted delay, and{" "}
              {d.heldout_comparison.stronger_baseline_minus_identified.ratio.toFixed(1)} times closer than that with per-joint offsets added.</p>
            <p className="small"><strong>Status: {d.status}{d.status === "proven" ? ", and it improves a public tool" : ""}.</strong>{" "}
              Rule: {d.status_rule}.{d.status === "proven" ? " Because the model it beats is the current public one, the correction improves a public tool (the cases page lists it that way)." : ""}</p>
            <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
              <a className="button" href="/astro/so_arm100_identified.xml" download><Download size={16} aria-hidden="true" /> Corrected MJCF</a>
              <a className="button secondary" href="/astro/so_arm100_identified.diff" download><Download size={16} aria-hidden="true" /> Diff against the public model</a>
            </div>
          </section>
          <section className="panel stack" aria-labelledby="rb-joints">
            <h2 id="rb-joints">Each joint&apos;s law, chosen on held-out episodes</h2>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Joint</th><th>Law</th><th>Parameters (95% interval)</th></tr></thead>
                <tbody>
                  {Object.entries(d.joints).map(([name, j]) => (
                    <tr key={name}>
                      <td data-label="Joint">{name.replace("_", " ")}</td>
                      <td data-label="Law">{j.title}</td>
                      <td data-label="Parameters">{Object.entries(j.params).map(([k, p]) => (
                        <div key={k}>{PARAM[k] ?? k}: {p.value.toPrecision(3)} ({p.ci95[0].toPrecision(3)} to {p.ci95[1].toPrecision(3)})</div>
                      ))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          {d.trace && (
            <section className="panel stack" aria-labelledby="rb-trace">
              <h2 id="rb-trace">The corrected simulator stays closer to the real arm</h2>
              <RobotTrace t={d.trace} />
            </section>
          )}
          <RobotPractice />
          <RobotFleet />
          <RobotPolicy />
          <RobotDrift />
          <section className="panel stack" aria-labelledby="rb-cmp">
            <h2 id="rb-cmp">Against the public model and fair baselines (held-out RMS, degrees)</h2>
            <div className="table-wrap">
              <table className="cards">
                <thead><tr><th>Model</th>{joints.map((j) => <th key={j}>{j.replace("_", " ")}</th>)}<th>Arm total</th></tr></thead>
                <tbody>
                  {MODELS.filter(([k]) => d.heldout[k]).map(([k, label]) => (
                    <tr key={k}>
                      <td data-label="Model">{label}</td>
                      {joints.map((j) => <td key={j} data-label={j} className="num">{d.heldout[k].joints[j]?.rms_deg.toFixed(2)}</td>)}
                      <td data-label="Arm total" className="num"><strong>{d.heldout[k].arm_total.rms_deg.toFixed(2)}</strong>
                        {" "}<span className="muted small">({d.heldout[k].arm_total.ci95[0].toFixed(2)} to {d.heldout[k].arm_total.ci95[1].toFixed(2)})</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted">Fair baseline: {d.fair_baseline.what} ({(d.fair_baseline.delay_s * 1000).toFixed(0)} ms). Stronger baseline:{" "}
              {d.stronger_baseline.what}. The gripper closes on objects the simulator scene does not contain, so it is shown but not in the arm total.</p>
          </section>
          <section className="panel stack" aria-labelledby="rb-data">
            <h2 id="rb-data">Data</h2>
            <p className="small">Recordings: {d.data.dataset} ({d.data.licence}), {d.data.episodes} episodes, {d.data.frames.toLocaleString()} frames
              at {d.data.fps} per second: {d.data.fit_episodes.length} episodes fit each law, {d.data.select_episodes.length} choose between laws, and{" "}
              {d.data.test_episodes.length} are held out for the comparison below, never used before it. Model: {d.model.source} ({d.model.licence}).</p>
          </section>
        </>
      )}
    </main>
  );
}
