"use client";

import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { RobotTrace, type Trace } from "./RobotTrace";

// The home page's impact beat: the SO-100 arm's simulator corrected from public recordings (scripts/robot_demo.py).
// Every number is read from the stored result (/astro/robot.json), so the page cannot say more than the result does.

interface Rms { rms_deg: number; ci95: [number, number] }
interface Robot {
  data: { dataset: string; episodes: number; test_episodes: number[] };
  heldout: Record<string, { arm_total: Rms }>;
  heldout_comparison: Record<string, { ratio: number }>;
  trace?: Trace;
}

export function RobotImpact() {
  const [d, setD] = useState<Robot | null>(null);
  const [fleet, setFleet] = useState<{ summary: { arms: number; pairs: number; own_mean_deg: number; borrowed_mean_deg: number;
    pairs_significantly_worse: number } } | null>(null);
  // Where the gripper ends up (scripts/robot_impact.py): the same replays, in millimetres at the hand.
  const [hand, setHand] = useState<{ gripper: Record<string, { rms_mm: { mean: number } }> } | null>(null);
  // A second robot (scripts/aloha_demo.py): the ALOHA against a model that was already identified.
  const [aloha, setAloha] = useState<{ heldout: Record<string, { arm_total: Rms }>; heldout_comparison: Record<string, { ratio: number }> } | null>(null);
  useEffect(() => {
    fetch("/astro/robot.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
    fetch("/astro/robot_fleet.json").then((r) => (r.ok ? r.json() : null)).then(setFleet).catch(() => setFleet(null));
    fetch("/astro/robot_impact.json").then((r) => (r.ok ? r.json() : null)).then(setHand).catch(() => setHand(null));
    fetch("/astro/robot_aloha.json").then((r) => (r.ok ? r.json() : null)).then(setAloha).catch(() => setAloha(null));
  }, []);
  if (!d) return null;
  const pub = d.heldout.default.arm_total, id = d.heldout.identified.arm_total;
  const ratio = d.heldout_comparison.default_minus_identified.ratio;
  return (
    <section className="band band-surface" aria-labelledby="robot-title">
      <div className="band-inner robot-impact">
        <div className="stack" style={{ gap: 14 }}>
          <h2 id="robot-title">Every cheap robot arm is slightly different from its simulator.</h2>
          <p>So policies trained in simulation fail on the real thing. PhysicsLens measures the arm itself: each joint&apos;s
            delay, lag and deadband, chosen by forecasts of motion it never saw, written into a corrected MuJoCo model.</p>
          <p>On the SO-100, the low-cost arm at the centre of Hugging Face&apos;s LeRobot community, from {d.data.episodes} public
            recordings: on the {d.data.test_episodes.length} held-out episodes the corrected simulator tracks the real joints to{" "}
            <strong className="num">{id.rms_deg.toFixed(2)}°</strong> against <strong className="num">{pub.rms_deg.toFixed(2)}°</strong> for the
            public model, {ratio.toFixed(1)} times closer.</p>
          {fleet && (
            <p>And it is per unit: on {fleet.summary.arms} owners&apos; arms, a calibration borrowed from another arm tracks worse than
              the arm&apos;s own in {fleet.summary.pairs_significantly_worse} of {fleet.summary.pairs} pairs
              ({fleet.summary.borrowed_mean_deg.toFixed(2)}° against {fleet.summary.own_mean_deg.toFixed(2)}°). Each arm and setup needs its own.</p>
          )}
          {aloha && (
            <p>And it is not one arm&apos;s trick: on the ALOHA arms the corrected simulator is{" "}
              {aloha.heldout_comparison.default_minus_identified.ratio.toFixed(1)} times closer to the real joints
              ({aloha.heldout.identified.arm_total.rms_deg.toFixed(2)}° against {aloha.heldout.default.arm_total.rms_deg.toFixed(2)}°) than
              MuJoCo Menagerie&apos;s model, which its authors had already identified on a real ALOHA.</p>
          )}
          <ul className="stat-strip count-strip">
            <li><span className="stat-number">{pub.rms_deg.toFixed(2)}°</span><span className="stat-label">public MuJoCo model, held-out error</span></li>
            <li><span className="stat-number">{id.rms_deg.toFixed(2)}°</span><span className="stat-label">corrected model, same episodes</span></li>
            {hand && (
              <>
                <li><span className="stat-number">{hand.gripper.default.rms_mm.mean.toFixed(1)} mm</span><span className="stat-label">public model, where it puts the hand</span></li>
                <li><span className="stat-number">{hand.gripper.identified.rms_mm.mean.toFixed(1)} mm</span><span className="stat-label">corrected model, at the hand</span></li>
              </>
            )}
          </ul>
          <p><a className="story-link" href="/robot">The corrected model, every joint&apos;s law and the download <ArrowRight size={15} aria-hidden="true" /></a></p>
        </div>
        {d.trace && <RobotTrace t={d.trace} />}
      </div>
    </section>
  );
}
