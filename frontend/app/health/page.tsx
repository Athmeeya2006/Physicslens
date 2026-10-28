"use client";

import { useEffect, useState } from "react";
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
    <main>
      <header className="page-head">
        <h1>Machine health</h1>
        <p className="tagline">
          Record the same part twice, weeks apart. PhysicsLens compares its natural frequency and damping and flags drift, an early sign of wear.
        </p>
      </header>

      {online === false && (
        <section className="panel">
          <h2>Needs the analysis server</h2>
          <p className="muted">
            The analysis server isn&apos;t running right now (it runs during judging sessions), so
            recordings can&apos;t be compared at the moment. With the server, you
            paste two result links of the same machine part; PhysicsLens refits both and flags a change in natural frequency
            or damping only when it exceeds 3 standard errors and a practical threshold (0.5% in frequency, 10% in damping),
            with a repeatability allowance for re-recording. In tests it raises no false alarm when the same system is
            re-recorded, and catches a 50% rise in damping and a 2% frequency shift.
          </p>
        </section>
      )}

      {online && <form className="panel upload" onSubmit={run}>
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
