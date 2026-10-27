"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

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

export default function HealthPage() {
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
      const r = await fetch(`${API_BASE}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseline: runId(baseline), current: runId(current) }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `${r.status}`);
      setResult(body as Comparison);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <header>
        <h1>Machine health</h1>
        <p className="tagline">
          Record the same part twice, weeks apart. PhysicsLens compares its natural frequency and damping and flags drift, an early sign of wear.
        </p>
        <p className="small"><a href="/">Analyze a new recording</a></p>
      </header>

      <form className="card upload" onSubmit={run}>
        <label>
          Baseline recording (result link or id)
          <input value={baseline} onChange={(e) => setBaseline(e.target.value)} placeholder="e.g. the analysis from last month" />
        </label>
        <label>
          Current recording (result link or id)
          <input value={current} onChange={(e) => setCurrent(e.target.value)} placeholder="the analysis from today" />
        </label>
        <button type="submit" disabled={busy || !baseline || !current}>{busy ? "Comparing (about 15 s)..." : "Compare dynamics"}</button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="card">
          <h2 className={`status ${result.drift ? "no_fit" : "accepted"}`}>{result.drift ? "Drift detected" : "No significant change"}</h2>
          <p>{result.summary}</p>
          <table>
            <thead>
              <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {result.metrics.map((m) => (
                <tr key={m.metric} className={m.flagged ? "flagged" : ""}>
                  <td>{m.label}</td>
                  <td>{fmt(m.baseline)} ± {fmt(m.baseline_std)} {m.unit}</td>
                  <td>{fmt(m.current)} ± {fmt(m.current_std)} {m.unit}</td>
                  <td>{m.change_pct >= 0 ? "+" : ""}{m.change_pct.toFixed(1)}%</td>
                  <td>{Math.abs(m.z).toFixed(1)} std. errors{m.flagged ? " (flagged)" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            A change is flagged only if it exceeds 3 standard errors and is large enough to matter (0.5% in frequency, 10% in damping).
            Uncertainties include a repeatability allowance for re-recording.
          </p>
        </section>
      )}
    </main>
  );
}
