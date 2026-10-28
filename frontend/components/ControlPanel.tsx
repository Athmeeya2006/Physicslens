"use client";

import { useEffect, useRef, useState } from "react";
import type { TrailStep } from "@/lib/api";
import { getControl, getHealth, startControl } from "@/lib/api";
import { AgentTrail } from "./AgentTrail";

// Muse Wonder writes a controller for the identified system; only MuJoCo runs on the identified model and its
// bootstrap variants decide whether it is accepted. Everything shown here is simulation, and is labeled so.

export type ControlGoal = "reach_and_hold" | "swing_up" | "stop" | "track_sine";

export interface ControlTask {
  family: "pendulum" | "oscillator" | "rotation";
  goal: ControlGoal;
  target: number;
  tolerance: number;
  actuator_limit: number;
  time_limit_s: number;
  frequency_hz: number;
}

interface Summary {
  successes: number; samples: number; success_rate: number; nominal_success: boolean; passed: boolean;
  mean_settle_s: number | null; mean_final_error: number | null; mean_peak_u: number | null; mean_rms_u: number | null;
  mean_saturation: number | null; errors: number;
}

interface Attempt {
  attempt: number; model: string; cached: boolean; approach: string | null; code: string | null;
  check: { ok: boolean; reason: string | null } | null; summary: Summary | null;
}

export interface ControlResult {
  task: ControlTask; label: string; samples: number; outcome: "accepted" | "failed" | "infeasible";
  message: string; infeasibility_note: string | null; accepted_attempt: number | null; best_attempt?: number | null;
  attempts: Attempt[]; trail: TrailStep[]; baseline: { summary: Summary; code: string };
}

export interface ControlView {
  task_id: string; status: "queued" | "running" | "done" | "error"; progress: string[]; error: string | null;
  result: ControlResult | null;
}

const GOALS: Record<string, { goal: ControlGoal; label: string }[]> = {
  pendulum: [
    { goal: "reach_and_hold", label: "Move to an angle and hold it" },
    { goal: "swing_up", label: "Swing up and balance upright" },
    { goal: "stop", label: "Stop the swing seen in the clip" },
    { goal: "track_sine", label: "Follow a sine" },
  ],
  oscillator: [
    { goal: "reach_and_hold", label: "Move to a position and hold it" },
    { goal: "stop", label: "Stop the motion seen in the clip" },
    { goal: "track_sine", label: "Follow a sine" },
  ],
};

const DEG = Math.PI / 180;

function num(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "n/a" : v.toFixed(digits);
}

export function ControlPanel({ runId, family, params, length }: {
  runId: string; family: string; params: Record<string, number>; length?: number;
}) {
  const goals = GOALS[family];
  const pendulum = family === "pendulum";
  // Defaults that a motor can actually do on this system: 1.5 times the torque or force the default target needs.
  const g = params.g ?? 9.81;
  const L = length ?? 1;
  const defaultLimit = pendulum ? 1.5 * g * L * Math.sin(30 * DEG) : 1.5 * (params.omega ?? 1) ** 2 * 0.05;
  const [goal, setGoal] = useState<ControlGoal>("reach_and_hold");
  const [target, setTarget] = useState(pendulum ? 30 : 5); // degrees, or cm from rest
  const [tolerance, setTolerance] = useState(pendulum ? 2 : 0.5);
  const [limit, setLimit] = useState(+defaultLimit.toPrecision(2));
  const [timeLimit, setTimeLimit] = useState(4);
  const [freq, setFreq] = useState(0.5);
  const [view, setView] = useState<ControlView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [allowed, setAllowed] = useState<boolean | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let live = true;
    // Public visitors don't get this panel: controller writing is for judges and development only.
    getHealth().then((h) => live && setAllowed(h.audience !== "public")).catch(() => live && setAllowed(false));
    return () => {
      live = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  if (!goals || !allowed) return null;

  const poll = (taskId: string) => {
    getControl(runId, taskId).then((v) => {
      setView(v);
      if (v.status === "queued" || v.status === "running") timer.current = setTimeout(() => poll(taskId), 1500);
    }).catch((e: Error) => setError(e.message));
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const unit = pendulum ? DEG : 0.01;
    const stop = goal === "stop";
    const task: ControlTask = {
      family: family as ControlTask["family"], goal, target: goal === "swing_up" || stop ? 0 : target * unit,
      // For "stop" the tolerance is a speed: degrees or centimeters per second.
      tolerance: tolerance * unit, actuator_limit: limit, time_limit_s: timeLimit, frequency_hz: freq,
    };
    startControl(runId, task).then((v) => {
      setView(v);
      if (v.status === "queued" || v.status === "running") poll(v.task_id);
    }).catch((err: Error) => setError(err.message));
  };

  const busy = view?.status === "queued" || view?.status === "running";
  const r = view?.result;
  const accepted = r?.attempts.find((a) => a.attempt === r.accepted_attempt);
  const best = accepted ?? r?.attempts.find((a) => a.attempt === r?.best_attempt);
  const angleUnit = pendulum ? "°" : " cm";

  return (
    <div className="panel">
      <h3>A controller for this system, written by Muse Wonder and tested in simulation</h3>
      <p className="sim-label" role="note">Simulation. The controller is tested on the identified model and on variants drawn from its uncertainty, in MuJoCo; never on the real object.</p>
      <p className="muted small" style={{ margin: "8px 0 12px" }}>
        Muse Wonder writes Python code for a motor {pendulum ? "at the pivot" : "on the spring axis"}. The code is checked, then run
        in an isolated process on the nominal model plus 50 bootstrap variants; it is accepted only if at least 90% of them,
        including the nominal one, meet the goal. Up to three attempts, each failure reported back to the model.
      </p>
      <form onSubmit={submit} className="control-form" style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
        <label className="small">Goal
          <select value={goal} onChange={(e) => setGoal(e.target.value as ControlGoal)}>
            {goals.map((o) => <option key={o.goal} value={o.goal}>{o.label}</option>)}
          </select>
        </label>
        {goal !== "swing_up" && goal !== "stop" && (
          <label className="small">{goal === "track_sine" ? "Amplitude" : "Target"} ({pendulum ? "degrees" : "cm from rest"})
            <input type="number" step="any" value={target} onChange={(e) => setTarget(+e.target.value)} />
          </label>
        )}
        <label className="small">Tolerance ({goal === "stop" ? (pendulum ? "degrees/s" : "cm/s") : pendulum ? "degrees" : "cm"})
          <input type="number" step="any" min={0} value={tolerance} onChange={(e) => setTolerance(+e.target.value)} />
        </label>
        <label className="small">Motor limit ({pendulum ? "N m" : "N"} per kg)
          <input type="number" step="any" min={0} value={limit} onChange={(e) => setLimit(+e.target.value)} />
        </label>
        <label className="small">Time limit (s)
          <input type="number" step="any" min={0.5} max={30} value={timeLimit} onChange={(e) => setTimeLimit(+e.target.value)} />
        </label>
        {goal === "track_sine" && (
          <label className="small">Frequency (Hz)
            <input type="number" step="any" min={0.05} max={5} value={freq} onChange={(e) => setFreq(+e.target.value)} />
          </label>
        )}
        <div style={{ alignSelf: "end" }}><button type="submit" disabled={busy}>{busy ? "Working..." : "Write and test a controller"}</button></div>
      </form>
      {error && <p className="error small">{error}</p>}
      {view?.status === "error" && <p className="error small">{view.error}</p>}
      {busy && <p className="muted small" style={{ marginTop: 8 }}>{view?.progress.at(-1) ?? "Queued"}...</p>}

      {r && (
        <div style={{ marginTop: 16 }}>
          <p><strong>{r.message}</strong></p>
          {r.infeasibility_note && r.outcome !== "infeasible" && <p className="note small">{r.infeasibility_note}</p>}
          {r.trail.length > 0 && <AgentTrail steps={r.trail} />}
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead><tr><th>Controller (simulation)</th><th>Variants that meet the goal</th><th>Mean settling time</th><th>Mean final error</th><th>Time at the motor limit</th></tr></thead>
              <tbody>
                {best?.summary && (
                  <tr>
                    <td>Muse Wonder&apos;s controller{accepted ? "" : ` (best of ${r.attempts.length} attempts, not accepted)`}</td>
                    <td className="num">{best.summary.successes} of {best.summary.samples}</td>
                    <td className="num">{best.summary.mean_settle_s == null ? "n/a" : `${num(best.summary.mean_settle_s)} s`}</td>
                    <td className="num">{pendulum ? `${num((best.summary.mean_final_error ?? NaN) / DEG)}°` : `${num((best.summary.mean_final_error ?? NaN) * 100)} cm`}</td>
                    <td className="num">{num((best.summary.mean_saturation ?? NaN) * 100, 0)}%</td>
                  </tr>
                )}
                <tr>
                  <td>Textbook controller (tuned on the textbook model)</td>
                  <td className="num">{r.baseline.summary.successes} of {r.baseline.summary.samples}</td>
                  <td className="num">{r.baseline.summary.mean_settle_s == null ? "n/a" : `${num(r.baseline.summary.mean_settle_s)} s`}</td>
                  <td className="num">{pendulum ? `${num((r.baseline.summary.mean_final_error ?? NaN) / DEG)}°` : `${num((r.baseline.summary.mean_final_error ?? NaN) * 100)} cm`}</td>
                  <td className="num">{num((r.baseline.summary.mean_saturation ?? NaN) * 100, 0)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="muted small" style={{ marginTop: 4 }}>
            Errors are {r.task.goal === "stop" ? `speeds (${pendulum ? "degrees" : "cm"} per second)` : `in ${angleUnit.trim()}`}, averaged over the last second.
          </p>
          {r.attempts.length > 0 && (
            <ol className="small" style={{ marginTop: 12, paddingLeft: 18 }}>
              {r.attempts.map((a) => (
                <li key={a.attempt}>
                  Attempt {a.attempt}:{" "}
                  {a.check && !a.check.ok ? `rejected before running (${a.check.reason})`
                    : a.summary ? `${a.summary.successes} of ${a.summary.samples} variants ${a.summary.passed ? "(accepted)" : "(rejected)"}` : "no result"}
                  {a.cached && <span className="muted"> (cached)</span>}
                  {a.approach && <div className="muted">{a.approach}</div>}
                </li>
              ))}
            </ol>
          )}
          {accepted?.code && (
            <details style={{ marginTop: 12 }}>
              <summary>Accepted controller code (Python)</summary>
              <pre style={{ overflowX: "auto", fontSize: "0.8rem" }}><code>{accepted.code}</code></pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
