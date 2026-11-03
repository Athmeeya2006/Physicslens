"use client";

import { useEffect, useState } from "react";
import { AgentOnCase } from "@/components/AgentOnCase";
import { TrendChart } from "@/components/TrendChart";
import { API_BASE, useBackend, formatUnit } from "@/lib/api";

interface MetricRow {
  metric: string;
  label: string;
  unit: string;
  baseline: number;
  baseline_std: number;
  current: number;
  current_std: number;
  change_pct: number;
  z: number;
  flagged: boolean;
}

interface Comparison {
  drift: boolean;
  summary: string;
  metrics: MetricRow[];
  durations_s: [number, number];
  runs: { baseline: { id: string; object?: string }; current: { id: string; object?: string } };
}

// The server answers JSON; anything else (an HTML error page, a proxy) becomes a readable message.
async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => null);
  if (!r.ok || data === null) throw new Error(data?.detail ?? "The analysis server didn't answer. Try again in a moment.");
  return data as T;
}

// Accept a pasted result link (".../?run=abc123") or a bare run id.
function runId(input: string): string {
  const trimmed = input.trim();
  try {
    return new URL(trimmed).searchParams.get("run") ?? trimmed;
  } catch {
    return trimmed;
  }
}

function fmt(v: number): string {
  return Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toPrecision(3);
}

interface Trend {
  drift: boolean;
  summary: string;
  recordings: { id: string; natural_freq_hz: [number, number]; decay_rate_per_s: [number, number]; flagged: Record<string, boolean> }[];
}

function TrendSection() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trend, setTrend] = useState<Trend | null>(null);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    const runs = text.split(/\s+/).map((l) => l.trim()).filter(Boolean).map(runId);
    setBusy(true);
    setError(null);
    setTrend(null);
    try {
      setTrend(await postJson<Trend>("/api/health/trend", { runs }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    // ?trend=id1,id2,... opens a trend directly (shareable link).
    const ids = new URLSearchParams(window.location.search).get("trend");
    if (ids) {
      setText(ids.split(",").join("\n"));
      setTimeout(() => (document.getElementById("trend-submit") as HTMLButtonElement | null)?.click(), 0);
    }
  }, []);

  const series = (key: "natural_freq_hz" | "decay_rate_per_s") =>
    trend?.recordings.map((rec, i) => ({ label: `#${i + 1}`, value: rec[key][0], std: rec[key][1], flagged: rec.flagged[key] })) ?? [];

  return (
    <section className="panel">
      <h3>Trend across recordings</h3>
      <form className="upload" onSubmit={run}>
        <label>
          3 to 12 result links or ids, oldest first, one per line
          <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <button id="trend-submit" type="submit" disabled={busy || !text.trim()}>{busy ? "Measuring each recording..." : "Show trend"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {trend && (
        <>
          <p className={trend.drift ? "error" : ""}><strong>{trend.summary}</strong></p>
          <div className="trend-grid">
            <TrendChart title="Natural frequency" unit="Hz" points={series("natural_freq_hz")} />
            <TrendChart title="Damping (amplitude decay rate)" unit="1/s" points={series("decay_rate_per_s")} />
          </div>
          <p className="muted small">Error bars: one standard error. Points marked drift differ significantly from recording #1.</p>
        </>
      )}
    </section>
  );
}


interface Share { k: number; n: number; wilson95: [number, number] }
interface PublicFaults {
  status: string;
  bearings: { faulty: { type_accuracy: Share }; healthy_false_alarms: Share; passed: boolean; rule?: string };
  spindown: { two_term_mean_sd: { a_rad_s2: [number, number]; b_per_s: [number, number] };
    published_mean_sd: { a_rad_s2: [number, number]; b_per_s: [number, number] }; agrees_with_published: boolean };
  sounds?: { fan_mean: Record<string, number>; pump_mean: Record<string, number> };
}

// Public recordings with known faults (scripts/health_public_demo.py), scored against their labels by rules fixed
// before scoring: what physics finds, and where it does not.
function PublicFaultsSection() {
  const [d, setD] = useState<PublicFaults | null>(null);
  useEffect(() => {
    fetch("/astro/health_public.json").then((r) => (r.ok ? r.json() : null)).then(setD).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const pct = (s: Share) => `${s.k} of ${s.n}`;
  const auc = (m?: Record<string, number>) => (m ? m.physics_auc.toFixed(2) : "n/a");
  return (
    <section className="panel stack" aria-labelledby="hp-title">
      <h2 id="hp-title">On public recordings of real machines with known faults</h2>
      <ul>
        <li><strong>Bearings</strong> (University of Ottawa run-to-failure data): from the bearing geometry and shaft rate, the
          physics rule names the fault type right in {pct(d.bearings.faulty.type_accuracy)} faulty recordings (chance is 1 in 4) and
          alarms on {pct(d.bearings.healthy_false_alarms)} healthy ones. {d.bearings.passed ? "It passes its own test." : "Bearing faults need more than this physics rule: it does not clear the bar set in advance, so they stay a research case, with the full numbers shown."}</li>
        <li><strong>A motor coasting down</strong>: Coulomb plus viscous friction, {d.spindown.two_term_mean_sd.a_rad_s2[0].toFixed(2)} rad/s² and{" "}
          {d.spindown.two_term_mean_sd.b_per_s[0].toFixed(4)} per second, against the published {d.spindown.published_mean_sd.a_rad_s2[0]} and{" "}
          {d.spindown.published_mean_sd.b_per_s[0]}: {d.spindown.agrees_with_published ? "it agrees" : "it does not agree"}.</li>
        {d.sounds && (
          <li><strong>Fan and pump sounds</strong> (MIMII): a physics score from the rotation rate and its harmonics separates abnormal from
            normal with an average area under the curve of {auc(d.sounds.fan_mean)} on fans and {auc(d.sounds.pump_mean)} on pumps; a generic
            learned detector on the same clips does better ({d.sounds.fan_mean.generic_template_auc.toFixed(2)} and{" "}
            {d.sounds.pump_mean.generic_template_auc.toFixed(2)}). Physics says which quantity moved; the learned detector separates better.</li>
        )}
      </ul>
      <p className="small muted">Every rule was fixed before scoring. Sources, licences and the full report: data/health_public/ in the repository.</p>
      <AgentOnCase file="agent_coastdown.json" what="the wheel's turning angle against time during one coast-down" />
    </section>
  );
}

export default function HealthPage() {
  const online = useBackend();
  const [baseline, setBaseline] = useState("");
  const [current, setCurrent] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Comparison | null>(null);

  useEffect(() => {
    const b = new URLSearchParams(window.location.search).get("baseline");
    if (b) setBaseline(b);
  }, []);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await postJson<Comparison>("/api/compare", { baseline: runId(baseline), current: runId(current) }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main id="main">
      <header className="page-head">
        <h1>Machine health</h1>
        <p className="tagline">
          Record the same part twice, weeks apart. PhysicsLens compares its natural frequency and damping and flags drift, an early sign of wear.
        </p>
      </header>

      <section className="panel stack">
        <h2>How it decides</h2>
        <ol>
          <li><strong>Two recordings of the same part.</strong> Analyze each on the Analyze page (a door closer, a spring
            mount, a pendulum-like arm), weeks apart, filmed the same way.</li>
          <li><strong>The same law, refitted.</strong> Each is fitted with the damped law of its kind, so damping is always
            measured (zero is a valid answer), and reduced to its natural frequency and decay rate, each with a bootstrap
            uncertainty.</li>
          <li><strong>A change must be both real and large enough to matter.</strong> It is flagged only beyond 3 standard
            errors <em>and</em> beyond a practical threshold: 0.5% in frequency, 10% in damping. A repeatability allowance
            covers what changes every time you re-record (the release, the temperature, the mounting).</li>
          <li><strong>A trend, not a snapshot.</strong> Up to 12 recordings show each quantity with its interval over time,
            so a slow drift is visible before it becomes a failure.</li>
        </ol>
        <h3>What it was tested on</h3>
        <p className="small">On simulated recordings with known changes (<code>tests/test_health.py</code>): no false alarm
          when the same system is recorded again, and both a 50% rise in damping and a 2% shift in frequency are flagged.
          On real machines with known faults the motor coast-down matches the published friction and bearing faults are a
          research case (below).</p>
      </section>

      <PublicFaultsSection />

      {online === false && (
        <section className="panel">
          <h2>Comparing your own recordings</h2>
          <p className="muted">
            Comparing recordings needs the analysis server, which is between sessions right now and restarts on its own
            within minutes. Comparing costs nothing: both recordings are already analyzed.
          </p>
        </section>
      )}

      {online && <form className="panel upload" onSubmit={run}>
        <p className="small muted">
          First analyze the same part twice on the Analyze page (for example a door closer, weeks apart). Each finished
          analysis has a link ending in ?run=...; paste the two links, or just the ids, here. Comparing costs nothing.
        </p>
        <label>
          Baseline recording (result link or id)
          <input value={baseline} onChange={(e) => setBaseline(e.target.value)} placeholder="e.g. the analysis from last month" />
        </label>
        <label>
          Current recording (result link or id)
          <input value={current} onChange={(e) => setCurrent(e.target.value)} placeholder="the analysis from today" />
        </label>
        <button type="submit" disabled={busy || !baseline || !current}>{busy ? "Comparing (about 15 s)..." : "Compare dynamics"}</button>
      </form>}

      {error && <p className="error">{error}</p>}

      {online && <TrendSection />}

      {result && (
        <section className="panel">
          <h2 className={`status ${result.drift ? "no_fit" : "accepted"}`}>{result.drift ? "Drift detected" : "No significant change"}</h2>
          <p>{result.summary}</p>
          <div className="table-wrap"><table>
            <thead>
              <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {result.metrics.map((m) => (
                <tr key={m.metric} className={m.flagged ? "flagged" : ""}>
                  <td>{m.label}</td>
                  <td>{fmt(m.baseline)} ± {fmt(m.baseline_std)} {formatUnit(m.unit)}</td>
                  <td>{fmt(m.current)} ± {fmt(m.current_std)} {formatUnit(m.unit)}</td>
                  <td>{m.change_pct >= 0 ? "+" : ""}{m.change_pct.toFixed(1)}%</td>
                  <td>{Math.abs(m.z).toFixed(1)} std. errors{m.flagged ? " (flagged)" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
          <p className="muted small">
            A change is flagged only if it exceeds 3 standard errors and is large enough to matter (0.5% in frequency, 10% in damping).
            Uncertainties include a repeatability allowance for re-recording.
          </p>
        </section>
      )}
    </main>
  );
}
