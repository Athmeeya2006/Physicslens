"use client";

import { Check, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { AuditCheck, Health, PhysicsAudit, RunView } from "@/lib/api";
import { formatUnit, getHealth, getRun, startRun, useBackend, withUncertainty } from "@/lib/api";

const POLL_MS = 1000;

const VERDICT: Record<PhysicsAudit["verdict"], { label: string; css: string }> = {
  consistent: { label: "Consistent with physics", css: "accepted" },
  violates_physics: { label: "Violates physics", css: "no_fit" },
  cant_judge: { label: "Can't judge", css: "ambiguous" },
};

const STATUS: Record<AuditCheck["status"], { label: string; css: string }> = {
  violation: { label: "violation", css: "no_fit" },
  pass: { label: "passed", css: "accepted" },
  skipped: { label: "not checked", css: "approximation" },
};

// The measured number as a reader writes it: "14.2 ± 0.6 m/s²", or a plain value when it has no uncertainty.
function measured(c: AuditCheck): string {
  if (c.value === null) return "";
  const unit = formatUnit(c.unit);
  if (c.std !== null && c.std > 0) {
    const [v, s] = withUncertainty(c.value, c.std);
    return `${v} ± ${s}${unit ? ` ${unit}` : ""}`;
  }
  return `${c.value.toPrecision(3)}${unit ? ` ${unit}` : ""}`;
}

function AuditView({ runId, audit }: { runId: string; audit: PhysicsAudit }) {
  const v = VERDICT[audit.verdict];
  return (
    <section className="panel stack" aria-labelledby="audit-verdict">
      <h2 id="audit-verdict"><span className={`verdict ${v.css}`}>{v.label}</span></h2>
      {audit.violations.length > 0 ? (
        <ul className="stack" style={{ gap: 8 }}>
          {audit.violations.map((c, i) => (
            <li key={i}><strong>{c.title}.</strong> {c.text}</li>
          ))}
        </ul>
      ) : (
        <p>{audit.headline}</p>
      )}
      {audit.checks.length > 0 && (
        <div className="table-wrap"><table>
          <thead>
            <tr><th>Check</th><th>Result</th><th>Measured</th><th>Significance</th><th>Details</th></tr>
          </thead>
          <tbody>
            {audit.checks.map((c, i) => (
              <tr key={i} className={c.status === "violation" ? "flagged" : ""}>
                <td>{c.title}</td>
                <td><span className={`verdict ${STATUS[c.status].css}`}>{STATUS[c.status].label}</span></td>
                <td>{measured(c)}</td>
                <td>{c.z !== null && c.status !== "skipped" ? `${Math.abs(c.z).toFixed(1)} std. errors` : ""}</td>
                <td className="small">{c.text}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
      )}
      <p className="muted small">
        A violation needs both at least 5 standard errors (8 for frame-to-frame jumps) and a size that matters in practice,
        such as g off by more than 25% or a bounce returning more than 110% of its speed. When the clip is too short,
        poorly tracked, has a variable frame rate or a moving camera, the audit says it can&apos;t judge instead of guessing.
      </p>
      <p className="small"><a href={`/?run=${runId}`}>Open the full analysis</a>: the equation, the ghost overlay and the decision trail.</p>
    </section>
  );
}

export default function AuditPage() {
  const online = useBackend();
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sizeCm, setSizeCm] = useState("");
  const [hint, setHint] = useState("");
  const [run, setRun] = useState<RunView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const turnstileBox = useRef<HTMLDivElement>(null);

  const poll = (id: string, failures = 0) => {
    getRun(id)
      .then((r) => {
        setRun(r);
        if (r.status === "queued" || r.status === "running") timer.current = setTimeout(() => poll(id), POLL_MS);
      })
      .catch((e: Error) => {
        if (failures < 5) timer.current = setTimeout(() => poll(id, failures + 1), POLL_MS * 2 ** (failures + 1));
        else setError(e.message);
      });
  };

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    // ?run=<id> shows the audit of an existing analysis, so a verdict can be shared by link.
    const shared = new URLSearchParams(window.location.search).get("run");
    if (shared) poll(shared);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  useEffect(() => {
    // Bot check for public uploads only, and only when the deployment configures it (as on the analyze page).
    const key = health?.turnstile_site_key;
    if (!key || !health?.can_upload || !turnstileBox.current) return;
    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
    script.async = true;
    script.onload = () => window.turnstile?.render(turnstileBox.current!, { sitekey: key, callback: setTurnstileToken });
    document.body.appendChild(script);
    return () => script.remove();
  }, [health]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setRun(null);
    try {
      const token = turnstileToken;
      setTurnstileToken(null);
      if (turnstileBox.current) window.turnstile?.reset(turnstileBox.current);
      const r = await startRun(file, { pendulumLengthM: null, objectSizeM: sizeCm ? Number(sizeCm) / 100 : null,
                                       slowMotion: null, objectHint: hint || null }, token, "/api/audit");
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
  const result = run?.status === "done" ? run.result : null;

  return (
    <main>
      <header className="page-head">
        <h1>Physics audit</h1>
        <p className="tagline">
          Does this motion obey physics? Upload a clip, real or generated, and PhysicsLens measures how far it strays:
          gravity, energy, a steady period, a law that forecasts it, and nothing that teleports or morphs.
        </p>
      </header>

      {online === null && (
        <p className="muted"><Loader2 size={14} className="spin" aria-hidden="true" /> Looking for the analysis server...</p>
      )}

      {online === false && !result && (
        <section className="panel">
          <h2>Needs the analysis server</h2>
          <p className="muted">
            The analysis server isn&apos;t running right now (it runs on a GPU notebook during judging sessions), so clips
            can&apos;t be audited at the moment. With the server, you upload a clip; SAM 3.1 tracks the moving object and the
            audit checks the track with plain numerical tests: g against 9.81 m/s² when something of known size is in view
            (or one g shared by every flight when nothing is), bounces that gain energy, a swing that grows or drifts in
            period, and jumps or shape changes between frames. Each violation is reported with its size and significance.
          </p>
        </section>
      )}

      {online && (
        <form className="panel upload" onSubmit={submit}>
          <label className={`dropzone ${file ? "has-file" : ""}`}>
            <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <Upload size={22} strokeWidth={1.8} aria-hidden="true" />
            <strong>{file ? file.name : "Choose a video to audit"}</strong>
            <span className="small muted">2 to 20 seconds, MP4 or MOV, camera held still</span>
          </label>
          <label className="field">
            What should it follow? Optional, for busy scenes
            <input type="text" maxLength={60} placeholder="e.g. the red ball" value={hint} onChange={(e) => setHint(e.target.value)} />
          </label>
          <label className="field">
            The moving object&apos;s longest side, cm (optional: lets g be checked against Earth&apos;s)
            <input type="number" inputMode="decimal" min="0.5" step="0.1" value={sizeCm} onChange={(e) => setSizeCm(e.target.value)} />
          </label>
          {health?.turnstile_site_key && <div ref={turnstileBox} />}
          <button type="submit" disabled={!file || busy || (health !== null && !health.can_upload)}>
            {busy ? <><Loader2 size={16} className="spin" aria-hidden="true" />Auditing</> : "Audit the motion"}
          </button>
          {health && !health.can_upload && <p className="muted small">Uploads are closed right now.</p>}
        </form>
      )}

      {error && <p className="error">{error}</p>}

      {run && (busy || run.status === "error") && (
        <ol className="progress panel" aria-live="polite">
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

      {result && run && (result.audit
        ? <AuditView runId={run.run_id} audit={result.audit} />
        : <p className="muted">This analysis was made before the physics audit existed; analyze the clip again to audit it.</p>)}
    </main>
  );
}
