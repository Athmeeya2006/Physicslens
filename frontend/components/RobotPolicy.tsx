"use client";

import { useEffect, useState } from "react";

// Calibrated on a human, tested on an AI (scripts/robot_policy.py, data/robot/policy.json): each arm's servo laws from
// its teleoperation, run on its owner's published ACT-policy rollouts. Shown only when the stored result exists;
// every number is read from it.

interface Iv { mean: number; ci95: [number, number] }
interface Arm { arm: string; rollouts: number; teleop_heldout: { own: Iv; baseline: Iv };
  policy: { own: Iv; baseline: Iv; borrowed_median: Iv }; policy_datasets: { dataset: string; licence: string }[] }
interface Retrained { arm: string; train_rollouts: number; test_rollouts: number; teleop_calibration: Iv; policy_calibration: Iv }
interface Policy {
  arms: Arm[];
  calibrated_on_policy?: Retrained[];
  summary: { arms: number; rollouts: number; rollouts_own_better_than_baseline: number; rollouts_own_better_than_borrowed: number };
  limits: string[];
}

const owner = (id: string) => id.split("__")[0];

export function RobotPolicy() {
  const [d, setD] = useState<Policy | null>(null);
  useEffect(() => {
    fetch("/astro/robot_policy.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const s = d.summary;
  const ratios = d.arms.map((a) => a.policy.baseline.mean / a.policy.own.mean);
  const owns = d.arms.map((a) => a.policy.own.mean), tele = d.arms.map((a) => a.teleop_heldout.own.mean);
  return (
    <section className="panel stack" aria-labelledby="rb-policy">
      <h2 id="rb-policy">Calibrated on a person, it predicts the arm under an AI policy{s.rollouts_own_better_than_baseline === s.rollouts ? `: closer in all ${s.rollouts} rollouts` : ""}</h2>
      <p>Every calibration here was fitted on teleoperation: a person moving a leader arm. {s.arms} of the owners also published their
        arm driven by a trained ACT policy. From the policy&apos;s commands alone, each arm&apos;s own calibration, never refitted, predicts
        where the arm went {Math.min(...ratios).toFixed(1)} to {Math.max(...ratios).toFixed(1)} times closer than a delay and offsets fitted to
        the same teleoperation, in <strong className="num">{s.rollouts_own_better_than_baseline} of {s.rollouts}</strong> rollouts, and closer
        than a calibration borrowed from another arm in {s.rollouts_own_better_than_borrowed} of {s.rollouts}.</p>
      <p>What it also shows: the policy drives the arm differently from a person, and the error grows from{" "}
        {Math.min(...tele).toFixed(2)}° to {Math.max(...tele).toFixed(2)}° on human driving to {Math.min(...owns).toFixed(2)}° to{" "}
        {Math.max(...owns).toFixed(2)}° under the policy.
        {d.calibrated_on_policy && d.calibrated_on_policy.length > 0 && <> Calibrating on the policy&apos;s own motion closes that: fitted on{" "}
          {d.calibrated_on_policy[0].train_rollouts} of an arm&apos;s policy rollouts, the calibration predicts the other{" "}
          {d.calibrated_on_policy[0].test_rollouts}{" "}
          {d.calibrated_on_policy.map((r) => `${(r.teleop_calibration.mean / r.policy_calibration.mean).toFixed(1)}`).join(" and ")} times closer than
          the teleoperation calibration ({d.calibrated_on_policy.map((r) => `${owner(r.arm)}: ${r.policy_calibration.mean.toFixed(2)}° against ${r.teleop_calibration.mean.toFixed(2)}°`).join("; ")}).</>}</p>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Arm (AI rollouts)</th><th className="num">Own calibration, human driving</th><th className="num">Own calibration, AI driving</th>
            <th className="num">Delay and offsets, AI driving</th><th className="num">Borrowed calibration, AI driving</th></tr></thead>
          <tbody>
            {d.arms.map((a) => (
              <tr key={a.arm}>
                <td data-label="Arm">{owner(a.arm)} ({a.rollouts})</td>
                <td data-label="Own, human" className="num">{a.teleop_heldout.own.mean.toFixed(2)}°</td>
                <td data-label="Own, AI" className="num"><strong>{a.policy.own.mean.toFixed(2)}°</strong></td>
                <td data-label="Delay and offsets, AI" className="num">{a.policy.baseline.mean.toFixed(2)}°</td>
                <td data-label="Borrowed, AI" className="num">{a.policy.borrowed_median.mean.toFixed(2)}°</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">RMS over the five arm joints per rollout, from the servo laws themselves (the fleet&apos;s evaluator).{" "}
        {d.limits.join("; ")}. Data: {d.arms.flatMap((a) => a.policy_datasets).map((p) => `${p.dataset} (${p.licence})`).join("; ")}.</p>
    </section>
  );
}
