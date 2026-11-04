"use client";

import { Download } from "lucide-react";
import { useEffect, useState } from "react";

// A second robot (scripts/aloha_demo.py, data/robot/aloha/result.json): the ALOHA arms against MuJoCo Menagerie's model,
// which its authors had already identified on a real ALOHA. Shown only when the stored result exists; every number
// is read from it.

interface Rms { rms_deg: number; ci95: [number, number] }
interface Aloha {
  status: string;
  data: { dataset: string; licence: string; episodes: number; test_episodes: number[] };
  heldout: Record<string, { joints: Record<string, Rms>; arm_total: Rms }>;
  heldout_comparison: Record<string, { improvement_deg: number; ci95: [number, number]; ratio: number }>;
}

const ROWS: [string, string][] = [
  ["default", "Menagerie ALOHA, as published (already identified on a real ALOHA)"],
  ["baseline_delay", "Menagerie + one fitted command delay"],
  ["baseline_delay_offset", "Menagerie + fitted delay and per-joint offsets"],
  ["identified", "Corrected by PhysicsLens"],
];

interface Transfer { tasks: { dataset: string; episodes: number; episodes_better: number; default: Rms; identified: Rms;
  default_minus_identified: { ratio: number } }[] }

export function RobotAloha() {
  const [d, setD] = useState<Aloha | null>(null);
  const [tr, setTr] = useState<Transfer | null>(null);
  useEffect(() => {
    fetch("/astro/robot_aloha.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
    fetch("/astro/robot_aloha_transfer.json").then((r) => (r.ok ? r.json() : null)).then(setTr).catch(() => setTr(null));
  }, []);
  if (!d) return null;
  const c = d.heldout_comparison.default_minus_identified;
  const joints = Object.keys(d.heldout.default.joints);
  const best = joints.reduce((a, b) => (d.heldout.default.joints[b].rms_deg - d.heldout.identified.joints[b].rms_deg
    > d.heldout.default.joints[a].rms_deg - d.heldout.identified.joints[a].rms_deg ? b : a));
  return (
    <section className="panel stack" aria-labelledby="rb-aloha">
      <h2 id="rb-aloha">A second robot: {c.ratio.toFixed(1)} times closer than a model that was already identified</h2>
      <p>The ALOHA&apos;s public MuJoCo model was not guessed: its authors fitted its gains, damping and friction to test motions on a
        real ALOHA. The same identification, run on {d.data.episodes} ordinary task recordings of another unit ({d.data.dataset}),
        tracks all twelve arm joints on {d.data.test_episodes.length} held-out episodes to{" "}
        <strong className="num">{d.heldout.identified.arm_total.rms_deg.toFixed(2)}°</strong> against{" "}
        {d.heldout.default.arm_total.rms_deg.toFixed(2)}° for the published model. The biggest gain is the{" "}
        {best.replace("/", " ").replace("_", " ")}: {d.heldout.default.joints[best].rms_deg.toFixed(2)}° to{" "}
        {d.heldout.identified.joints[best].rms_deg.toFixed(2)}°.</p>
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Simulator</th><th className="num">Held-out error, twelve arm joints</th></tr></thead>
          <tbody>
            {ROWS.filter(([k]) => d.heldout[k]).map(([k, label]) => (
              <tr key={k}>
                <td data-label="Simulator">{k === "identified" ? <strong>{label}</strong> : label}</td>
                <td data-label="Held-out error" className="num">{k === "identified" ? <strong>{d.heldout[k].arm_total.rms_deg.toFixed(2)}°</strong>
                  : `${d.heldout[k].arm_total.rms_deg.toFixed(2)}°`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {tr && (
        <>
          <p><strong>It carries to the robot&apos;s other tasks.</strong> Identified on the coffee task and refitted on nothing, the correction
            beats the published model on {tr.tasks.reduce((a, t) => a + t.episodes_better, 0)} of{" "}
            {tr.tasks.reduce((a, t) => a + t.episodes, 0)} episodes of three other tasks.</p>
          <div className="table-wrap">
            <table className="cards">
              <thead><tr><th>Task</th><th className="num">Menagerie</th><th className="num">Corrected on coffee</th><th className="num">Episodes closer</th></tr></thead>
              <tbody>
                {tr.tasks.map((t) => (
                  <tr key={t.dataset}>
                    <td data-label="Task">{t.dataset.split("_static_")[1]}</td>
                    <td data-label="Menagerie" className="num">{t.default.rms_deg.toFixed(2)}°</td>
                    <td data-label="Corrected on coffee" className="num"><strong>{t.identified.rms_deg.toFixed(2)}°</strong></td>
                    <td data-label="Episodes closer" className="num">{t.episodes_better} of {t.episodes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
        <a className="button secondary" href="/astro/aloha_identified.xml" download><Download size={16} aria-hidden="true" /> Corrected ALOHA MJCF</a>
      </div>
      <p className="small muted">Derived from MuJoCo Menagerie&apos;s aloha model, BSD-3-Clause, Copyright (c) 2023 Trossen Robotics
        (<a href="/astro/LICENSE-menagerie-aloha.txt">licence</a>). Data: {d.data.dataset} ({d.data.licence}).</p>
    </section>
  );
}
