"use client";

import { Check, Download, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AgentTrail } from "@/components/AgentTrail";
import { VoiceNotes } from "@/components/VoiceNotes";
import { Tex } from "@/components/Equation";
import type { GalleryItem, RunView, Study, Sweep } from "@/lib/api";
import { addStudyRun, createStudy, formatUnit, getGallery, withUncertainty, getRun, getStudy, posterUrl, reviewStudy, studyFileUrl, useBackend, useHealth } from "@/lib/api";

const EXAMPLES = [
  "Are the front shock absorbers on my car worn out?",
  "I need the damping of this cabinet door hinge for my robot simulation.",
  "Is the bearing in this desk fan wearing out over the next month?",
];

const DELIVERABLE = { simulation_model: "Simulation model", health_check: "Wear check", parameter_sheet: "Parameter sheet" };

function setStudyParam(id: string) {
  const url = new URL(window.location.href);
  url.searchParams.set("study", id);
  window.history.replaceState(null, "", url);
}

export default function StudyPage() {
  const online = useBackend();
  const health = useHealth();
  const judgesOnly = !!health && health.audience === "public"; // the server refuses studies without the access code
  const [goal, setGoal] = useState("");
  const [study, setStudy] = useState<Study | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<RunView | null>(null);
  const [results, setResults] = useState<Record<string, RunView>>({});
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("");
  const [setting, setSetting] = useState("");
  const [lengthCm, setLengthCm] = useState("");
  const [sizeCm, setSizeCm] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [published, setPublished] = useState<GalleryItem[]>([]);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("study");
    if (id) getStudy(id).then(setStudy).catch((e: Error) => setError(e.message));
    else getGallery().then((g) => setPublished(g.filter((x) => x.kind === "study"))).catch(() => undefined);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  // Each recording's verdict, for the list; finished results never change, so fetch each once.
  useEffect(() => {
    for (const r of study?.recordings ?? []) {
      if (results[r.run_id]?.status === "done" || results[r.run_id]?.status === "error") continue;
      getRun(r.run_id).then((v) => setResults((m) => ({ ...m, [r.run_id]: v }))).catch(() => undefined);
    }
  }, [study]); // eslint-disable-line react-hooks/exhaustive-deps

  const plan = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy("The agent is planning the study (open-weight reasoning model)");
    try {
      const s = await createStudy(goal);
      setStudy(s);
      setStudyParam(s.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const next = study ? nextRecording(study) : null;

  const poll = (id: string, studyId: string) => {
    getRun(id).then(async (r) => {
      setRun(r);
      if (r.status === "queued" || r.status === "running") {
        timer.current = setTimeout(() => poll(id, studyId), 2000);
        return;
      }
      setResults((m) => ({ ...m, [id]: r }));
      setBusy("The agent is reviewing the study");
      try {
        setStudy(await reviewStudy(studyId));
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setBusy(null);
        setRun(null);
      }
    }).catch(() => (timer.current = setTimeout(() => poll(id, studyId), 4000)));
  };

  const upload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!study || !file) return;
    setError(null);
    try {
      const out = await addStudyRun(study.id, file, label || next?.label || "", {
        pendulumLengthM: lengthCm ? Number(lengthCm) / 100 : null, objectSizeM: sizeCm ? Number(sizeCm) / 100 : null,
        slowMotion: null, objectHint: null, setting: setting ? Number(setting) : null,
      });
      setStudy(out.study);
      setFile(null);
      setLabel("");
      setSetting("");
      poll(out.run.run_id, study.id);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const verdictOf = (runId: string) => results[runId]?.result?.status ?? results[runId]?.status;
  const analyzing = !!run && (run.status === "queued" || run.status === "running");

  return (
    <main className="stack" style={{ gap: 28 }}>
      <header className="page-head">
        <h1>Studies</h1>
        <p className="tagline">
          Give a goal, not a clip. The agent plans what to film, asks for recordings until the question is answered, and
          combines them into one deliverable. Every recording is still verified on its own, and the combined numbers come
          from math, not from the model.
        </p>
      </header>

      {!study && (
        <section className="panel stack">
          <h2>Start a study</h2>
          {online === false ? (
            <p className="muted">
              The analysis server isn&apos;t running right now (it runs during judging sessions), so a new
              study can&apos;t start at the moment.
            </p>
          ) : judgesOnly ? (
            <p className="muted">
              Starting a study is open to judges with the access code: each one plans recordings and calls the models.
              The finished results on the Analyze page show the same agent at work.
            </p>
          ) : (
            <form className="stack" onSubmit={plan}>
              <label className="field">
                What do you want to find out?
                <textarea rows={3} maxLength={400} value={goal} onChange={(e) => setGoal(e.target.value)}
                          placeholder="e.g. Are the front shock absorbers on my car worn out?" />
              </label>
              <div className="chips">
                {EXAMPLES.map((x) => <button type="button" key={x} className="chip" onClick={() => setGoal(x)}>{x}</button>)}
              </div>
              <button type="submit" disabled={!online || goal.trim().length < 8 || !!busy}>
                {busy ? <><Loader2 size={16} className="spin" aria-hidden="true" />Planning</> : "Plan the study"}
              </button>
            </form>
          )}
        </section>
      )}

      {!study && published.length > 0 && (
        <section className="stack">
          <h2>Finished studies</h2>
          <ul className="gallery-list">
            {published.map((it) => (
              <li key={it.id}>
                <a className="gallery-item" href={`/study?study=${it.id}`}>
                  <img src={posterUrl(it.id)} alt="" loading="lazy" />
                  <span>
                    <span className="g-title">{it.title}</span>
                    <span className="g-desc" style={{ display: "block" }}>{it.description}</span>
                    <span className="g-meta"><span>{it.credit}</span></span>
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}

      {busy && <p className="muted"><Loader2 size={14} className="spin" aria-hidden="true" /> {busy}</p>}
      {error && <p className="error">{error}</p>}

      {study && (
        <>
          <section className="result-head">
            <h2 className={study.status === "finished" ? "accepted" : "ambiguous"}>
              {study.status === "finished" ? "Study complete" : "Study in progress"}
            </h2>
            <p className="reason">{study.goal}</p>
          </section>

          <div className="result-grid">
            <div className="stack">
              <section className="panel">
                <h3>The plan: {DELIVERABLE[study.plan.deliverable]}</h3>
                <p className="small muted" style={{ marginBottom: 8 }}>Measure {study.plan.quantities.join("; ")}. Done when {lowerFirst(study.plan.done_when)}</p>
                <ol className="checklist">
                  {study.plan.recordings.map((p) => {
                    const done = study.recordings.some((r) => r.label === p.label);
                    return (
                      <li key={p.label} className={done ? "done" : ""}>
                        {done ? <Check size={14} aria-hidden="true" /> : <span className="dot" aria-hidden="true" />}
                        <span><strong>{p.label}</strong> {p.how_to_film}</span>
                      </li>
                    );
                  })}
                </ol>
              </section>

              {study.status !== "finished" && online && (
                <section className="panel stack">
                  <h3>{next ? `Next: ${next.label}` : "Add a recording"}</h3>
                  {next && <p>{next.how_to_film}</p>}
                  {analyzing ? (
                    <ol className="progress" aria-live="polite">
                      {run!.progress.map((p, i) => (
                        <li key={i} className={i === run!.progress.length - 1 ? "active" : "done"}>
                          {i === run!.progress.length - 1 ? <Loader2 size={14} className="spin" aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}
                          {p}
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <form className="stack" onSubmit={upload}>
                      <label className={`dropzone ${file ? "has-file" : ""}`}>
                        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                        <Upload size={20} strokeWidth={1.8} aria-hidden="true" />
                        <strong>{file ? file.name : "Choose the recording"}</strong>
                      </label>
                      <VoiceNotes onNotes={(n) => {
                        if (n.label) setLabel(n.label);
                        if (n.pendulum_length_m) setLengthCm(String(Math.round(n.pendulum_length_m * 1000) / 10));
                        if (n.object_size_m) setSizeCm(String(Math.round(n.object_size_m * 1000) / 10));
                      }} />
                      <div className="scale-fields">
                        <label className="field">Label<input value={label} placeholder={next?.label ?? "Recording"} onChange={(e) => setLabel(e.target.value)} /></label>
                        {study.plan.sweep && (
                          <label className="field">
                            {study.plan.sweep.variable}{study.plan.sweep.unit ? `, ${study.plan.sweep.unit}` : ""}
                            <input type="number" min="0" step="any" value={setting}
                              placeholder={String(study.plan.sweep.values[study.recordings.length] ?? "")}
                              onChange={(e) => setSetting(e.target.value)} />
                          </label>
                        )}
                        <label className="field">Pendulum length, cm<input type="number" min="0.5" step="0.1" value={lengthCm} onChange={(e) => setLengthCm(e.target.value)} /></label>
                        <label className="field">Object size, cm<input type="number" min="0.5" step="0.1" value={sizeCm} onChange={(e) => setSizeCm(e.target.value)} /></label>
                      </div>
                      <button type="submit" disabled={!file || !!busy}>Analyze and add to the study</button>
                    </form>
                  )}
                </section>
              )}

              <section className="panel">
                <h3>Recordings</h3>
                {study.recordings.length === 0 ? <p className="muted small">None yet.</p> : (
                  <ul className="recordings">
                    {study.recordings.map((r) => {
                      const v = verdictOf(r.run_id);
                      return (
                        <li key={r.run_id}>
                          <span className={`verdict ${v === "done" || v === "running" || v === "queued" ? "" : v ?? ""}`}>{VERDICT[v ?? ""] ?? "Analyzing"}</span>
                          <a href={`/?run=${r.run_id}`}>{r.label}</a>
                          <span className="muted small">{results[r.run_id]?.result?.equation?.title ?? ""}</span>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </section>

              {study.sweep && <SweepPanel sweep={study.sweep} />}

              {study.combined.length > 0 && (
                <section className="panel">
                  <h3>Combined over the recordings</h3>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>Quantity</th><th>Value</th><th>Recordings</th><th>Agreement</th></tr></thead>
                      <tbody>
                        {study.combined.map((c) => (
                          <tr key={c.name}>
                            <td><Tex tex={c.latex} /></td>
                            <td className="num">{withUncertainty(c.value, c.std).join(" ± ")} {formatUnit(c.unit)}{c.uncalibrated && <span className="uncal">uncalibrated</span>}</td>
                            <td className="num">{c.n}</td>
                            <td>{c.n === 1 ? <span className="muted small">one recording</span>
                              : <span className={`verdict ${c.consistent ? "accepted" : "rejected"}`}>{c.consistent ? "Agree" : "Disagree"}</span>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="muted small" style={{ marginTop: 8 }}>
                    Weighted by each recording&apos;s uncertainty. &ldquo;Disagree&rdquo; means the recordings differ by more than
                    their error bars allow (chi-squared test, p below 0.01): something changed, or one measurement is off.
                  </p>
                </section>
              )}

              {study.health && (
                <section className="panel">
                  <h3>Wear check</h3>
                  <p><span className={`verdict ${study.health.drift ? "rejected" : "accepted"}`}>{study.health.drift ? "Drift" : "No significant change"}</span> {study.health.summary}</p>
                </section>
              )}

              {study.turns.length > 0 && (
                <section className="panel">
                  <h3>Agent&apos;s conclusion so far</h3>
                  <p>{study.turns.at(-1)!.message}</p>
                  {study.turns.at(-1)!.overruled && <p className="note">{study.turns.at(-1)!.overruled}</p>}
                  {study.turns.at(-1)!.maintenance_note && <p style={{ marginTop: 8 }}><strong>Maintenance:</strong> {study.turns.at(-1)!.maintenance_note}</p>}
                  {study.turns.at(-1)!.next_check && <p><strong>Next check:</strong> {study.turns.at(-1)!.next_check}</p>}
                </section>
              )}

              <section className="panel">
                <h3>Deliverables</h3>
                <div className="export-links">
                  <a className="button" href={studyFileUrl(study.id, "report")} download><Download size={16} aria-hidden="true" />Study report</a>
                  {study.plan.deliverable === "simulation_model" && study.status === "finished" && (
                    <a className="button secondary" href={studyFileUrl(study.id, "mjcf")} download>MuJoCo model (combined)</a>
                  )}
                </div>
              </section>
            </div>
            <div className="panel">
              <h3>Decision trail</h3>
              <AgentTrail steps={study.trail} />
            </div>
          </div>
        </>
      )}
    </main>
  );
}

const VERDICT: Record<string, string> = {
  accepted: "Verified", ambiguous: "Tie", no_fit: "No fit", error: "Failed",
};

function nextRecording(s: Study) {
  const last = s.turns.at(-1);
  if (last?.action === "record_more" && last.next_recording) return last.next_recording;
  return s.plan.recordings.find((p) => !s.recordings.some((r) => r.label === p.label)) ?? null;
}

function lowerFirst(text: string): string {
  return text && /^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text;
}

function SweepPanel({ sweep }: { sweep: Sweep }) {
  const unit = sweep.unit ? ` ${sweep.unit}` : "";
  const fmt = (v: number) => (Math.abs(v) >= 100 || Math.abs(v) < 0.01 ? v.toPrecision(3) : v.toFixed(3));
  return (
    <section className="panel stack">
      <h3>{sweep.response} against {sweep.variable}</h3>
      <p className="small muted">
        The agent expected: <strong>{sweep.expected_law}</strong> ({sweep.why}). Code fits every candidate law and keeps
        the one that predicts each recording from the others.
      </p>
      {sweep.points.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Recording</th><th>{sweep.variable}</th><th>{sweep.response}</th></tr></thead>
            <tbody>
              {sweep.points.map((p) => (
                <tr key={p.run_id}><td><a href={`/?run=${p.run_id}`}>{p.label}</a></td><td>{fmt(p.x)}{unit}</td>
                  <td>{fmt(p.y)} ± {fmt(p.std)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p><strong>{sweep.winner ? sweep.winner.title : sweep.status === "too_few" ? "Not enough settings yet" : "No single law yet"}</strong>: {sweep.reason}</p>
      {sweep.next?.y !== undefined && (
        <p className="prediction">
          Prediction before filming: at {sweep.variable} = {fmt(sweep.next.x)}{unit}, {sweep.response} ={" "}
          {fmt(sweep.next.y)} (95%: {fmt(sweep.next.low!)} to {fmt(sweep.next.high!)}).
        </p>
      )}
      {sweep.next?.separates && (
        <p className="prediction">Record at {sweep.variable} = {fmt(sweep.next.x)}{unit} to separate {sweep.next.separates.join(" and ")}.</p>
      )}
      {sweep.checks.map((c) => (
        <p key={c.label} className={c.within_95 ? "check ok" : "check miss"}>
          {c.label}: predicted {fmt(c.predicted)}, measured {fmt(c.measured)} ({c.z >= 0 ? "+" : ""}{c.z.toFixed(1)} sigma),{" "}
          {c.within_95 ? "inside the 95% interval" : "outside the 95% interval"}.
        </p>
      ))}
    </section>
  );
}
