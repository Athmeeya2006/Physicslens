"use client";

import { ArrowLeft, Check, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Gallery } from "@/components/Gallery";
import { VoiceNotes } from "@/components/VoiceNotes";
import { Hero } from "@/components/Hero";
import { ResultView } from "@/components/ResultView";
import type { Budget, GalleryItem, Health, RunView } from "@/lib/api";
import { getBudget, getGallery, getHealth, getRun, NEXT_SESSION_UTC, startRun, useBackend } from "@/lib/api";

declare global {
  interface Window {
    turnstile?: {
      render: (el: HTMLElement, opts: { sitekey: string; callback: (token: string) => void }) => void;
      reset: (el?: HTMLElement) => void;
    };
  }
}

const POLL_MS = 1000;
const REASONING_NAME: Record<string, string> = {
  baseten: "Muse Spark Open (open weights)",
  glimmer: "Muse Glimmer 30B (open weights)",
  together: "Muse Glimmer 30B (open weights)",
};

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const online = useBackend();
  const [unreachable, setUnreachable] = useState(false);
  const offline = online === false || unreachable;
  const [budget, setBudget] = useState<Budget | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [lengthCm, setLengthCm] = useState("");
  const [sizeCm, setSizeCm] = useState("");
  const [slowMo, setSlowMo] = useState("");
  const [hint, setHint] = useState("");
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
      .catch(() => setUnreachable(true));
    getGallery().then(setGallery).catch(() => setGallery([]));
    // ?run=<id> opens a finished run, so results can be reloaded and shared by link.
    const shared = new URLSearchParams(window.location.search).get("run");
    if (shared) poll(shared);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const poll = (id: string, failures = 0) => {
    getRun(id)
      .then((r) => {
        setRun(r);
        if (r.status === "queued" || r.status === "running") timer.current = setTimeout(() => poll(id), POLL_MS);
      })
      .catch((e: Error) => {
        // A dropped request on a phone shouldn't end a two-minute analysis: retry with backoff first.
        if (failures < 5) timer.current = setTimeout(() => poll(id, failures + 1), POLL_MS * 2 ** (failures + 1));
        else setError(e.message);
      });
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
      const size = sizeCm ? Number(sizeCm) / 100 : null;
      const token = turnstileToken;
      // A bot-check token is single use: ask for a fresh one before the next upload.
      setTurnstileToken(null);
      if (turnstileBox.current) window.turnstile?.reset(turnstileBox.current);
      const r = await startRun(file, { pendulumLengthM: length, objectSizeM: size,
                                       slowMotion: slowMo ? Number(slowMo) : null, objectHint: hint || null }, token);
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
          <Hero />
        )}

        <div className="home-grid">
          <section className="panel stack" aria-labelledby="analyze-title">
            <h2 id="analyze-title">Analyze a recording</h2>
            {health && !offline && (
              <p className="server-badge small">
                <span className="dot" aria-hidden="true" /> Analysis server online
                {health.reasoning && <>, reasoning on {REASONING_NAME[health.reasoning.provider] ?? health.reasoning.model}</>}
              </p>
            )}
            {online === null ? (
              <p className="muted"><Loader2 size={14} className="spin" aria-hidden="true" /> Looking for the analysis server...</p>
            ) : offline ? (
              <div className="stack" style={{ gap: 14 }}>
                <ol className="how">
                  <li><strong>Film it.</strong> Anything that moves, from a phone held still: a swing, a bounce, a spin-down.</li>
                  <li><strong>Meta&apos;s models measure and propose.</strong> SAM 3.1 tracks the object; Muse Spark Open, Meta&apos;s
                    open-weight model, reads the scene and proposes equations, even ones it writes itself.</li>
                  <li><strong>Math decides.</strong> Each equation must forecast motion it never saw; the winner comes with
                    uncertainties, a ghost on the video and a MuJoCo model.</li>
                </ol>
                {NEXT_SESSION_UTC && Date.parse(NEXT_SESSION_UTC) > Date.now() && (
                  <p className="small"><strong>Next live session:</strong> {new Date(NEXT_SESSION_UTC).toUTCString().replace(":00 GMT", " UTC")}
                    {" "}({new Date(NEXT_SESSION_UTC).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })} your time).</p>
                )}
                <p className="muted small">
                  The analysis server runs during judging sessions and isn&apos;t running right now.
                  Every recording on this page is a complete, real analysis; a <a href="/study">study</a> shows the agent
                  planning several recordings toward one goal.
                </p>
              </div>
            ) : health && !health.can_upload ? (
              <p className="muted">
                Analyzing your own clip is open to judges with the access code (each analysis calls the models). Every
                recording in the list is a complete, real analysis: open one to see the agent&apos;s decision trail.
              </p>
            ) : (
              <form className="stack" onSubmit={submit}>
                <label className={`dropzone ${file ? "has-file" : ""}`}>
                  <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                  <Upload size={22} strokeWidth={1.8} aria-hidden="true" />
                  <strong>{file ? file.name : "Choose or record a video"}</strong>
                  <span className="small muted">2 to 20 seconds, MP4 or MOV; longer clips are trimmed</span>
                </label>
                <VoiceNotes onNotes={(n) => {
                  if (n.pendulum_length_m) setLengthCm(String(Math.round(n.pendulum_length_m * 1000) / 10));
                  if (n.object_size_m) setSizeCm(String(Math.round(n.object_size_m * 1000) / 10));
                  if (n.object_hint) setHint(n.object_hint);
                  if (n.slow_motion) setSlowMo(String(n.slow_motion));
                }} />
                <label className="field">
                  What should it measure? Optional, for busy scenes
                  <input type="text" maxLength={60} placeholder="e.g. the red ball" value={hint} onChange={(e) => setHint(e.target.value)} />
                </label>
                <fieldset className="scale-fields">
                  <legend>Scale, optional: only needed when nothing of known size is in view</legend>
                  <label className="field">
                    Pendulum length, cm
                    <input type="number" inputMode="decimal" min="0.5" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} />
                  </label>
                  <label className="field">
                    Or the moving object&apos;s longest side, cm
                    <input type="number" inputMode="decimal" min="0.5" step="0.1" value={sizeCm} onChange={(e) => setSizeCm(e.target.value)} />
                  </label>
                </fieldset>
                <label className="field">
                  Slow-motion clip? How many times slower it plays (8 for 240 fps shown at 30), else leave empty
                  <input type="number" inputMode="decimal" min="1" step="any" value={slowMo} onChange={(e) => setSlowMo(e.target.value)} />
                </label>
                {health?.turnstile_site_key && <div ref={turnstileBox} />}
                <button type="submit" disabled={!file || busy || !canUpload}>
                  {busy ? <><Loader2 size={16} className="spin" aria-hidden="true" />Analyzing</> : "Analyze the motion"}
                </button>
                <ul className="tips">
                  <li>Hold the phone still, side-on to the motion.</li>
                  <li>Lay an A4 sheet flat in the plane of motion for real-world scale.</li>
                  <li>Frequency, damping and friction are exact even without a scale reference.</li>
                  <li>Anything that moves works: swinging, bouncing, thrown, rolling, sliding or braking.</li>
                </ul>
                <p className="small muted">
                  Your clip is sent to Meta&apos;s Model API for tracking, and kept with its result so the result link keeps
                  working. Don&apos;t upload anything private.
                </p>
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
