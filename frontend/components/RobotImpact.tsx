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
  useEffect(() => {
    fetch("/astro/robot.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
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
          <ul className="stat-strip count-strip">
            <li><span className="stat-number">{pub.rms_deg.toFixed(2)}°</span><span className="stat-label">public MuJoCo model, held-out error</span></li>
            <li><span className="stat-number">{id.rms_deg.toFixed(2)}°</span><span className="stat-label">corrected model, same episodes</span></li>
          </ul>
          <p><a className="story-link" href="/robot">The corrected model, every joint&apos;s law and the download <ArrowRight size={15} aria-hidden="true" /></a></p>
        </div>
        {d.trace && <RobotTrace t={d.trace} />}
      </div>
    </section>
  );
}
