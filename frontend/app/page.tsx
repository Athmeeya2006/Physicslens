"use client";

import { useEffect, useRef, useState } from "react";
import { ResultView } from "@/components/ResultView";
import type { Budget, Health, RunView } from "@/lib/api";
import { getBudget, getHealth, getRun, startRun } from "@/lib/api";

const POLL_MS = 1000;

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [lengthCm, setLengthCm] = useState("");
  const [run, setRun] = useState<RunView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const accessCode = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("code");

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        if (h.app_env === "dev") getBudget().then(setBudget).catch(() => setBudget(null));
      })
      .catch(() => setError("Cannot reach the PhysicsLens backend."));
    // ?run=<id> opens a finished run, so results can be reloaded and shared by link.
    const shared = new URLSearchParams(window.location.search).get("run");
    if (shared) poll(shared);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const poll = (id: string) => {
    getRun(id)
      .then((r) => {
        setRun(r);
        if (r.status === "queued" || r.status === "running") {
          timer.current = setTimeout(() => poll(id), POLL_MS);
        } else if (health?.app_env === "dev") {
          getBudget().then(setBudget).catch(() => undefined);
        }
      })
      .catch((e: Error) => setError(e.message));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setRun(null);
    try {
      const length = lengthCm ? Number(lengthCm) / 100 : null;
      const r = await startRun(file, length, accessCode);
      setRun(r);
      const url = new URL(window.location.href);
      url.searchParams.set("run", r.run_id);
      window.history.replaceState(null, "", url);
      poll(r.run_id);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const busy = run?.status === "queued" || run?.status === "running";

  return (
    <main>
      {health?.mock_mode && <div className="mock-banner">MOCK MODE: model responses are fake (tests only)</div>}
      <header>
        <h1>PhysicsLens</h1>
        <p className="tagline">Film something moving. Get the equation that governs it, verified.</p>
        {budget && (
          <div className="budget" title="Dev only: spend vs caps">
            {Object.entries(budget).map(([p, b]) => (
              <span key={p}>
                {p}: ${b.spent_usd.toFixed(3)} / ${b.hard_cap_usd.toFixed(0)}
              </span>
            ))}
          </div>
        )}
      </header>

      <form className="card upload" onSubmit={submit}>
        <label className="file">
          <input type="file" accept="video/*" capture="environment" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          {file ? file.name : "Choose or record a video (2 to 20 s)"}
        </label>
        <label>
          Pendulum length in cm (optional, if no A4 sheet is in view)
          <input type="number" inputMode="decimal" min="1" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} />
        </label>
        <button type="submit" disabled={!file || busy}>{busy ? "Analyzing..." : "Analyze motion"}</button>
        <p className="muted small">
          Tips: hold the phone still and side-on to the motion, and keep an A4 sheet flat in the same plane for scale.
        </p>
      </form>

      {error && <p className="error">{error}</p>}

      {run && (busy || run.status === "error") && (
        <ol className="card progress">
          {run.progress.map((p, i) => (
            <li key={i} className={i === run.progress.length - 1 && busy ? "active" : ""}>{p}</li>
          ))}
          {run.status === "error" && <li className="error">{run.error}</li>}
        </ol>
      )}

      {run?.status === "done" && run.result && <ResultView runId={run.run_id} result={run.result} />}
    </main>
  );
}
