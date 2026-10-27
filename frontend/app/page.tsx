"use client";

import { useEffect, useRef, useState } from "react";
import { Gallery } from "@/components/Gallery";
import { ResultView } from "@/components/ResultView";
import type { Budget, GalleryItem, Health, RunView } from "@/lib/api";
import { accessCode as readAccessCode, getBudget, getGallery, getHealth, getRun, startRun } from "@/lib/api";

declare global {
  interface Window {
    turnstile?: { render: (el: HTMLElement, opts: { sitekey: string; callback: (token: string) => void }) => void };
  }
}

const POLL_MS = 1000;

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [lengthCm, setLengthCm] = useState("");
  const [run, setRun] = useState<RunView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const turnstileBox = useRef<HTMLDivElement>(null);

  const accessCode = readAccessCode();
  const withCode = (path: string) => (accessCode ? `${path}${path.includes("?") ? "&" : "?"}code=${encodeURIComponent(accessCode)}` : path);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        if (h.app_env === "dev") getBudget().then(setBudget).catch(() => setBudget(null));
      })
      .catch(() => setError("Cannot reach the PhysicsLens backend."));
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
        if (r.status === "queued" || r.status === "running") {
          timer.current = setTimeout(() => poll(id), POLL_MS);
        } else if (health?.app_env === "dev") {
          getBudget().then(setBudget).catch(() => undefined);
        }
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

  const openRun = (id: string) => {
    if (timer.current) clearTimeout(timer.current);
    const url = new URL(window.location.href);
    url.searchParams.set("run", id);
    window.history.replaceState(null, "", url);
    setError(null);
    poll(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
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
        <nav className="small nav">
          {health?.can_go_live && <a href={withCode("/live")}>Go live with the camera</a>}
          <a href={withCode("/health")}>Machine health</a>
        </nav>
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

      {health && !health.can_upload && (
        <p className="card muted">Uploading your own clip is closed right now. Explore the gallery below: every result there is a real analysis.</p>
      )}
      {(!health || health.can_upload) && (
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
        {health?.turnstile_site_key && <div ref={turnstileBox} />}
      </form>
      )}

      {error && <p className="error">{error}</p>}

      {run && (busy || run.status === "error") && (
        <ol className="card progress">
          {run.progress.map((p, i) => (
            <li key={i} className={i === run.progress.length - 1 && busy ? "active" : ""}>{p}</li>
          ))}
          {run.status === "error" && <li className="error">{run.error}</li>}
        </ol>
      )}

      {run?.status === "done" && run.result && (
        <>
          {gallery.find((g) => g.id === run.run_id) && (
            <p className="credit">{gallery.find((g) => g.id === run.run_id)!.credit}</p>
          )}
          <ResultView runId={run.run_id} result={run.result} />
        </>
      )}

      <Gallery items={gallery} onOpen={openRun} />
    </main>
  );
}
