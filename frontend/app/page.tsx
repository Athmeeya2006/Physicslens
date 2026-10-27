"use client";

import { ArrowLeft, Check, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Gallery } from "@/components/Gallery";
import { ResultView } from "@/components/ResultView";
import type { Budget, GalleryItem, Health, RunView } from "@/lib/api";
import { getBudget, getGallery, getHealth, getRun, HAS_BACKEND, startRun } from "@/lib/api";

declare global {
  interface Window {
    turnstile?: { render: (el: HTMLElement, opts: { sitekey: string; callback: (token: string) => void }) => void };
  }
}

const POLL_MS = 1000;

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(!HAS_BACKEND);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [lengthCm, setLengthCm] = useState("");
  const [run, setRun] = useState<RunView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const turnstileBox = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        if (h.app_env === "dev") getBudget().then(setBudget).catch(() => setBudget(null));
      })
      .catch(() => setOffline(true));
    getGallery().then(setGallery).catch(() => setGallery([]));
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
        if (r.status === "queued" || r.status === "running") timer.current = setTimeout(() => poll(id), POLL_MS);
      })
      .catch((e: Error) => setError(e.message));
  };

  useEffect(() => {
    // Bot check for public uploads only, and only when the deployment configures it.
    const key = health?.turnstile_site_key;
    if (!key || !health?.can_upload || !turnstileBox.current) return;
    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
    script.async = true;
    script.onload = () => window.turnstile?.render(turnstileBox.current!, { sitekey: key, callback: setTurnstileToken });
    document.body.appendChild(script);
    return () => script.remove();
  }, [health]);

  const setRunParam = (id: string | null) => {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("run", id);
    else url.searchParams.delete("run");
    window.history.replaceState(null, "", url);
  };

  const openRun = (id: string) => {
    if (timer.current) clearTimeout(timer.current);
    setRunParam(id);
    setError(null);
    poll(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const closeRun = () => {
    if (timer.current) clearTimeout(timer.current);
    setRun(null);
    setRunParam(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setRun(null);
    try {
      const length = lengthCm ? Number(lengthCm) / 100 : null;
      const r = await startRun(file, length, turnstileToken);
      setRun(r);
      setRunParam(r.run_id);
      poll(r.run_id);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const busy = run?.status === "queued" || run?.status === "running";
  const done = run?.status === "done" && run.result;
  const canUpload = !offline && (health?.can_upload ?? false);

  return (
    <>
      {health?.mock_mode && <div className="mock-banner">Mock mode: model responses are fake (automated tests only)</div>}
      <main className="stack" style={{ gap: 32 }}>
        {done ? (
          <div className="stack">
            <p><button className="secondary" onClick={closeRun}><ArrowLeft size={16} aria-hidden="true" />All recordings</button></p>
            {gallery.find((g) => g.id === run.run_id) && <p className="credit">{gallery.find((g) => g.id === run.run_id)!.credit}</p>}
            <ResultView runId={run.run_id} result={run.result!} />
          </div>
        ) : (
          <div className="home-head">
            <h1>Film something moving. Get its equation, verified.</h1>
            <p>
              SAM 3.1 tracks the object, an agent proposes candidate equations, and numerical simulation keeps only the one that
              forecasts motion it never saw. Export the result to MuJoCo, or compare recordings over weeks to catch wear.
            </p>
          </div>
        )}

        <div className="home-grid">
          <section className="panel stack" aria-labelledby="analyze-title">
            <h2 id="analyze-title">Analyze a recording</h2>
            {offline ? (
              <p className="muted">
                The analysis server isn&apos;t connected to this preview, so new clips can&apos;t be analyzed here. Every recording
                in the list is a complete, real analysis you can open.
              </p>
            ) : health && !health.can_upload ? (
              <p className="muted">Analyzing your own clip is closed right now. Every recording in the list is a complete, real analysis.</p>
            ) : (
              <form className="stack" onSubmit={submit}>
                <label className={`dropzone ${file ? "has-file" : ""}`}>
                  <input type="file" accept="video/*" capture="environment" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                  <Upload size={22} strokeWidth={1.8} aria-hidden="true" />
                  <strong>{file ? file.name : "Choose or record a video"}</strong>
                  <span className="small muted">2 to 20 seconds, MP4 or MOV</span>
                </label>
                <label className="field">
                  Pendulum length in cm (optional, when nothing of known size is in view)
                  <input type="number" inputMode="decimal" min="1" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} />
                </label>
                {health?.turnstile_site_key && <div ref={turnstileBox} />}
                <button type="submit" disabled={!file || busy || !canUpload}>
                  {busy ? <><Loader2 size={16} className="spin" aria-hidden="true" />Analyzing</> : "Analyze the motion"}
                </button>
                <ul className="tips">
                  <li>Hold the phone still, side-on to the motion.</li>
                  <li>Lay an A4 sheet flat in the plane of motion for real-world scale.</li>
                  <li>Frequency, damping and friction are exact even without a scale reference.</li>
                </ul>
              </form>
            )}
            {error && <p className="error">{error}</p>}
            {run && (busy || run.status === "error") && (
              <ol className="progress" aria-live="polite">
                {run.progress.map((p, i) => {
                  const active = i === run.progress.length - 1 && busy;
                  return (
                    <li key={i} className={active ? "active" : "done"}>
                      {active ? <Loader2 size={14} className="spin" aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}
                      {p}
                    </li>
                  );
                })}
                {run.status === "error" && <li className="error">{run.error}</li>}
              </ol>
            )}
            {budget && (
              <p className="small muted">
                Development spend: {Object.entries(budget).map(([p, b]) => `${p} ${b.spent_usd.toFixed(3)} of ${b.hard_cap_usd.toFixed(0)}`).join(", ")}
              </p>
            )}
          </section>
          <Gallery items={gallery} onOpen={openRun} activeId={run?.run_id} />
        </div>
      </main>
    </>
  );
}
