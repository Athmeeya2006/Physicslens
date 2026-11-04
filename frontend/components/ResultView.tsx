"use client";

import { Boxes, Clock, Download, FileCode2, FileJson, FileText, Film, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import type { AnalysisResult, TrackPoint } from "@/lib/api";
import { clipUrl, type ExportKind, exportUrl, isGallery, isOpenWeight, makeShareVideo, useBackend, formatUnit } from "@/lib/api";
import { answering, FOLLOWUPS } from "@/lib/followups";
import { AgentReasoning } from "./AgentReasoning";
import { AgentTrail, useLaws } from "./AgentTrail";
import { AudioPanel } from "./AudioPanel";
import { ControlPanel } from "./ControlPanel";
import { DepthPanel } from "./DepthPanel";
import { Law, Tex } from "./Equation";
import { ForecastChart } from "./ForecastChart";
import { OutcomeCard, ReadableLaw } from "./OutcomeCard";
import { StatusChip } from "./StatusChip";
import { type Prediction, PredictionChart } from "./PredictionChart";
import { GhostPlayer } from "./GhostPlayer";
import { SimToReal } from "./SimToReal";
import { Term } from "./Term";
import { WhatIf } from "./WhatIf";

const STATUS_TEXT: Record<AnalysisResult["status"], string> = {
  accepted: "Verified equation",
  ambiguous: "Needs one more experiment",
  no_fit: "No clean governing equation",
};

const SUPERSCRIPT: Record<string, string> = {
  "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
};
const pow10 = (e: number) => `× 10${[...String(e)].map((c) => SUPERSCRIPT[c] ?? c).join("")}`;

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) {
    // 2.96 × 10⁻⁴ rather than 2.96e-4: the page is read by people, not parsers.
    const e = Math.floor(Math.log10(a));
    return `${(v / 10 ** e).toFixed(2)} ${pow10(e)}`;
  }
  return v.toPrecision(4);
}

/** Shared power of ten for a value, its uncertainty and interval: 0 unless the value is tiny or huge. */
function sciExponent(v: number): number {
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? Math.floor(Math.log10(a)) : 0;
}

// Two significant figures of the uncertainty decide how many decimals the value deserves.
function decimalsFor(std: number): number | null {
  if (!(std > 0) || !Number.isFinite(std)) return null;
  return Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(std))));
}

function fmtAt(v: number, d: number | null): string {
  const a = Math.abs(v);
  if (d === null || (a !== 0 && a >= 1e5)) return fmt(v);
  return v.toFixed(d);
}

// Units that carry a length depend on the scale (m, m/s^2, 1/m); angles, times and pure numbers don't.
function needsScale(unit: string): boolean {
  return /(^|[^a-z])m($|[^a-z])/.test(unit);
}

// Without a scale the nominal 1000 px/m makes a "mm" one pixel, so say pixels.
function mujocoGap(m: number, uncalibrated: boolean): string {
  const v = m * 1000;
  const unit = uncalibrated ? "px" : "mm";
  return v < 0.001 ? `less than 0.001 ${unit}` : `${v.toFixed(v < 1 ? 3 : 1)} ${unit}`;
}

function refereeWon(r: AnalysisResult): boolean {
  const sk = r.skeptic;
  return !!sk && "tested" in sk && !sk.survived && sk.verdict_after?.winner === r.hypothesis_votes?.winner;
}

// Readable names for the tags; the exact model id stays in the tooltip for anyone verifying.
function modelName(id: string): string {
  if (id.startsWith("meta-models/muse-spark-open")) return "Muse Spark Open · open weights";
  if (id.startsWith("muse-glimmer")) return "Muse Glimmer 30B · open weights";
  if (id === "sam-3.1") return "SAM 3.1";
  if (id.startsWith("muse-spark")) return `Muse Spark 1.3${id.endsWith("contributor") ? " (development)" : ""}`;
  return id;
}

const SCALE_SOURCE: Record<string, string> = {
  user: "the length you gave", object_size: "the object's size you gave", gravity: "Earth's gravity (assumed)",
  known_scale: "how the video was drawn",
};

function prettyModel(id: string, titles: Record<string, string>): string {
  return titles[id] ?? id.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}

// Every part of a result is on the page, in this order, with jump links at the top: nothing hides behind a tab.
const RESULT_TABS = ["trail", "forecast", "laws", "exports", "details"];
const TAB_LABEL: Record<string, string> = { trail: "Decision trail", forecast: "Forecast", laws: "Every law compared",
  exports: "Use the result", details: "Measurement details" };
const SECTION_TITLE: Record<string, string> = {trail: "Decision trail: what the agent and the math did", forecast: "Forecast: does the law predict motion it never saw?", laws: "The law and every law compared", exports: "Use the result: exports, controller and what-ifs", details: "Measurement details"};

// Where the held-out stretch falls in the clip: the forecast's own split mapped onto the ghost's frames, which span
// exactly the fitted segment (that segment need not be the whole clip).
function heldoutSpan(r: AnalysisResult): [number, number] | null {
  const f = r.forecast;
  const g = r.ghosts?.[0]?.points;
  if (!f || !g?.length || !r.clip.frames || f.t.length < 2) return null;
  const [f0, f1] = [g[0][0], g[g.length - 1][0]];
  const t0 = f.t[0], t1 = f.t[f.t.length - 1];
  if (!(t1 > t0) || !(f1 > f0)) return null;
  const start = f0 + ((f.heldout_start - t0) / (t1 - t0)) * (f1 - f0);
  return [Math.max(0, start / r.clip.frames), Math.min(1, f1 / r.clip.frames)];
}

export function ResultView({ runId, result, credit }: { runId: string; result: AnalysisResult; credit?: string }) {
  // Gallery orbit results carry a prediction file (scripts/halley_demo.py and orbit_demo.py prediction).
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  useEffect(() => {
    if (!isGallery(runId) || !result.prediction_available) return;
    fetch(`/gallery/${runId}/prediction.json`).then((r) => (r.ok ? r.json() : null)).then(setPrediction)
      .catch(() => setPrediction(null));  // most gallery results have none: nothing to show
  }, [runId, result.prediction_available]);
  const r = result;
  const [whatIfGhost, setWhatIfGhost] = useState<{ label: string; points: TrackPoint[] } | null>(null);
  const [share, setShare] = useState<{ state: "idle" | "busy" | "done" | "error"; url?: string; error?: string }>({ state: "idle" });
  const titles = Object.fromEntries(r.ranking.map((f) => [f.model_id, f.title]));
  const laws = useLaws();
  const used = Array.from(new Set((r.trail ?? []).map((t) => t.model).filter((m): m is string => !!m && m !== "simulate-and-compare")));
  // Model ids get a model tag; any other trail entry names a numerical method and is tagged as math.
  const isModel = (m: string) => /^(meta-models\/|muse-|sam-)/.test(m);
  const models = used.filter(isModel);
  const methods = used.filter((m) => !isModel(m));
  const online = useBackend();
  const canAsk = online === true || !isGallery(runId);
  const bestBic = Math.min(...r.ranking.map((f) => f.bic));

  // Models in blue with a sparkle, the math in grey with a clock; the exact model id stays in the tooltip.
  const modelTags = (
    <ul className="model-tags">
      {models.map((m) => (
        <li key={m}><span className={`model-tag${isOpenWeight(m) ? " open" : ""}`} title={m}><Sparkles size={13} aria-hidden="true" />{modelName(m)}</span></li>
      ))}
      {["numerical verification", ...methods].map((m) => (
        <li key={m}><span className="model-tag math"><Clock size={13} aria-hidden="true" />{m}</span></li>
      ))}
    </ul>
  );

  const onShare = () => {
    setShare({ state: "busy" });
    makeShareVideo(runId)
      .then((url) => setShare({ state: "done", url }))
      .catch((e: Error) => setShare({ state: "error", error: e.message }));
  };

  return (
    <section className="stack" aria-labelledby="result-title">
      <header className="result-hero">
        <div className="chip-row">
          <StatusChip status={r.status} label={STATUS_TEXT[r.status]} />
        </div>
        {answering(runId) ? (
          <>
            <h1 id="result-title">{answering(runId)!.headline}</h1>
            {r.outcome && (
              <p className="tagline">{r.status !== "accepted" && <strong>Still open: </strong>}{formatUnit(r.outcome.headline)}</p>
            )}
          </>
        ) : (
          <h1 id="result-title">{r.outcome ? formatUnit(r.outcome.headline) : STATUS_TEXT[r.status]}</h1>
        )}
        {credit && <p className="credit">{credit}</p>}
        {FOLLOWUPS[runId] && (
          <p className="followup-note"><strong>The experiment the agent asked for was run.</strong> {FOLLOWUPS[runId].settled}{" "}
            <a href={`/?run=${FOLLOWUPS[runId].answer}`}>See that result</a></p>
        )}
        {answering(runId) && (
          <p className="followup-note"><strong>This recording answers the agent&apos;s request</strong> from an earlier analysis of the same
            setup. {answering(runId)!.settled}{" "}<a href={`/?run=${answering(runId)!.question}`}>See the request</a></p>
        )}
      </header>
      <div className="result-top">
        <div>
        {r.video_kept === false ? (
          <div className="panel"><p className="note">A saved live session: the video was not kept, only the track and
            the numbers. The equation, the exports and the what-ifs below work as for any result.</p></div>
        ) : <GhostPlayer
          src={clipUrl(runId)}
          width={r.clip.width}
          height={r.clip.height}
          fps={r.clip.fps}
          real={r.real_track_px}
          secondReal={r.double?.outer_track_px}
          objectSizePx={r.object_size_px}
          whatIf={whatIfGhost}
          mujoco={r.mujoco?.points}
          ghosts={(r.ghosts ?? []).map((g) => ({ ...g, model_id: prettyModel(g.model_id, titles) }))}
          heldout={heldoutSpan(r)}
        />}
        </div>
        <div className="stack" style={{ gap: 14 }}>
          {r.outcome ? (
            <OutcomeCard o={r.outcome} exportHref={(k) => exportUrl(runId, k as ExportKind)} split tie={r.status === "ambiguous"}
                         extra={<><dt>Models used</dt><dd>{modelTags}</dd></>} />
          ) : (
            <dl className="outcome-facts"><dt>Verdict</dt><dd>{r.reason}</dd><dt>Models used</dt><dd>{modelTags}</dd></dl>
          )}
          {r.equation && (
            <div className="top-law">
              <p className="small">{r.status === "accepted" ? "The verified law" : "The leading law (not yet decided)"}: {r.equation.title}</p>
              <div className="law-scroll"><Tex tex={r.outcome?.readable?.latex_numeric ?? r.equation.latex} display /></div>
            </div>
          )}
          {r.scale_note && <p className="note">{r.scale_note}</p>}
        </div>
      </div>

      <AgentReasoning steps={r.trail ?? []} />

      <nav className="on-page" aria-label="On this page">
        {RESULT_TABS.map((id) => <a key={id} href={`#${id}`}>{TAB_LABEL[id]}</a>)}
      </nav>
        <section id="trail" className="result-section" aria-labelledby="trail-h">
          <h2 id="trail-h">{SECTION_TITLE["trail"]}</h2>
          <div className="stack">
            <div className="result-head">
              <h3 className={r.status}>{STATUS_TEXT[r.status]}</h3>
              <p className="reason">{r.reason}</p>
              {r.hypothesis_votes && (
                <p className="small muted">
                  {r.hypothesis_votes.proposed_by > 0
                    ? `Proposed by ${r.hypothesis_votes.proposed_by} of ${r.hypothesis_votes.samples} independent samples of the reasoning model before the math checked it.`
                    : refereeWon(r)
                      ? `None of ${r.hypothesis_votes.samples} independent samples proposed this law: the referee did, and it passed the held-out test and fit better by BIC.`
                      : `None of ${r.hypothesis_votes.samples} independent samples proposed this law; the agent tested it later or it is a baseline the math always tests.`}
                </p>
              )}
              {r.skeptic && "tested" in r.skeptic && r.skeptic.survived && r.skeptic.tested > 0 && (
                <p className="survived-badge">
                  Survived {r.skeptic.tested} challenge{r.skeptic.tested === 1 ? "" : "s"} from a referee model: no alternative law
                  did better by the same rule (forecast the held-out motion, then the best BIC).
                </p>
              )}
            </div>
            <AgentTrail steps={r.trail ?? []} runId={runId} guideStored={r.setup_guide_available}
              latex={Object.fromEntries(r.ranking.filter((f) => f.latex).map((f) => [f.model_id, f.latex!]))} />
          </div>
        </section>
        <section id="forecast" className="result-section" aria-labelledby="forecast-h">
          <h2 id="forecast-h">{SECTION_TITLE["forecast"]}</h2>
          <div className="stack">
            {!r.forecast && !prediction && <p className="muted">No held-out forecast for this result.</p>}
      {r.forecast && (
        <div className="panel forecast">
          <h3>Forecast check: does the model predict motion it never saw?</h3>
          <ForecastChart f={r.forecast} uncalibrated={!!r.scale_note} units={r.units} />
        </div>
      )}
      {prediction && (
        <div className="panel">
          <h3>The prediction, years ahead: the law alone, and with the planets</h3>
          <PredictionChart p={prediction} />
        </div>
      )}
      {/* No textbook baseline for scaled units (an orbit in AU and days) or for a law the agent wrote for free planar
          motion: the server builds none, so none is asked for. */}
      {r.status === "accepted" && r.fit_state && !r.units && !(r.family === "plane" && String(r.fit_state.model_id ?? "").startsWith("custom_")) &&
        <SimToReal runId={runId} />}
          </div>
        </section>
        <section id="laws" className="result-section" aria-labelledby="laws-h">
          <h2 id="laws-h">{SECTION_TITLE["laws"]}</h2>
          <div className="stack">
      {r.equation && (
        <div className="panel">
          <h3>{r.equation.title}</h3>
          <div className="equation"><Tex tex={r.equation.latex} display /></div>
          {r.outcome?.readable && <ReadableLaw r={r.outcome.readable} />}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Parameter</th><th>Value</th><th className="ci-col">95% interval</th></tr></thead>
              <tbody>
                {r.parameters?.map((p) => {
                  const ex = sciExponent(p.value);
                  const k = 10 ** ex;
                  const d = decimalsFor(p.std / k);
                  const sci = ex ? ` ${pow10(ex)}` : "";
                  const at = (v: number) => (ex ? (v / k).toFixed(d ?? 3) : fmtAt(v, d));
                  const ci = `${at(p.ci95[0])} to ${at(p.ci95[1])}${sci}`;
                  return (
                    <tr key={p.name}>
                      <td><Tex tex={p.latex} /> {p.derived && <span className="muted small">{p.derived}</span>}</td>
                      <td className="num"><span className="param-value">{at(p.value)}</span> ± {at(p.std)}{sci} {formatUnit(p.unit)}{" "}
                        {p.uncalibrated && <span className="uncal">uncalibrated</span>}
                        <div className="ci-inline muted small">95%: {ci}</div></td>
                      <td className="num ci-col">{ci}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {r.bootstrap_reps ? <p className="muted small">Every interval from {r.bootstrap_reps} bootstrap resamples.</p> : null}
          {r.identifiability?.map((e) => {
            const tex = (n: string) => r.parameters?.find((p) => p.name === n)?.latex ?? n;
            return (
              <p key={e.params.join("-")} className="note">
                <Tex tex={tex(e.params[0])} /> and <Tex tex={tex(e.params[1])} /> move together in the refits (correlation{" "}
                {e.corr.toFixed(2)}): this recording pins down their combination better than each one. {e.advice}
              </p>
            );
          })}
        </div>
      )}
      {r.double && (
        <div className="panel">
          <h3>Double pendulum: {r.double.chaotic ? "chaotic" : "regular"} motion</h3>
          <div className="equation"><Tex tex={r.double.latex} display /></div>
          <p className="num">
            <Tex tex="g" /> = {r.double.params.g.toFixed(3)} m/s², mass ratio <Tex tex="\mu" /> = {r.double.params.mu.toFixed(2)},{" "}
            <Tex tex="L_1" /> = {r.double.L1.toFixed(3)} m, <Tex tex="L_2" /> = {r.double.L2.toFixed(3)} m
            {r.scale_note && <span className="uncal">uncalibrated</span>}
          </p>
          <p style={{ marginTop: 8 }}>
            {r.double.chaotic && r.double.predictability_s
              ? `Nearby futures separate e-fold every ${(1 / r.double.lyapunov_per_s).toFixed(2)} s (Lyapunov exponent ${r.double.lyapunov_per_s.toFixed(1)}/s). With this tracking noise the motion can be forecast for about ${r.double.predictability_s.toFixed(1)} s, and no further.`
              : `Lyapunov exponent ${r.double.lyapunov_per_s.toFixed(2)}/s: small differences stay small, so the motion stays predictable.`}
          </p>
          <p className="muted small" style={{ marginTop: 8 }}>
            Fitted on short predictions restarted from the measured state, because a single long simulation of a chaotic
            system always drifts away. The third ghost is one prediction from the start, to show that drift.
          </p>
        </div>
      )}
      {r.bounces && (
        <div className="panel">
          <h3>Bounces: {r.bounces.accepted ? "hybrid model verified" : "hybrid model not confirmed"}</h3>
          <div className="equation"><Tex tex={r.bounces.latex} display /></div>
          <p className="num">
            <Tex tex="g" /> = {r.bounces.g.toFixed(3)} ± {r.bounces.g_std.toFixed(3)} m/s²
            {r.scale_note && <span className="uncal">uncalibrated</span>}, coefficient of restitution{" "}
            <Tex tex="e" /> = {r.bounces.e_mean.toFixed(3)} ± {r.bounces.e_std.toFixed(3)}
          </p>
          <div className="table-wrap" style={{ marginTop: 10 }}>
            <table>
              <thead><tr><th>Bounce</th><th>Time</th><th>Restitution e</th><th>Horizontal speed kept</th></tr></thead>
              <tbody>
                {r.bounces.impacts.map((imp, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td><td className="num">{imp.t.toFixed(2)} s</td>
                    <td className="num">{imp.e.toFixed(3)} ± {imp.e_std.toFixed(3)}</td><td className="num">{(imp.tangential * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            One g is shared by every flight.
            {r.bounces.forecast_nrmse != null && ` Later bounces are predicted from the first ones within ${(r.bounces.forecast_nrmse * 100).toFixed(1)}%.`}
            {r.bounces.settled_at ? ` After ${r.bounces.settled_at.toFixed(1)} s the hops were too small to resolve.` : ""}
          </p>
          {r.bounces.mujoco_contact && (
            <p className="small" style={{ marginTop: 8 }}>
              MuJoCo contact model: damping ratio {r.bounces.mujoco_contact.dampratio.toFixed(3)} (time constant{" "}
              {r.bounces.mujoco_contact.timeconst} s), tuned inside MuJoCo until its ball keeps e ={" "}
              {r.bounces.mujoco_contact.e_mujoco.toFixed(3)} of its speed, as measured.{" "}
              <a href={exportUrl(runId, "contact")} download>Download the contact model</a>
            </p>
          )}
        </div>
      )}
      <div className="panel">
        <h3>Every candidate tested</h3>
        <div className="table-wrap">
          <table className="cards">
            <thead><tr><th>Model, and why it was tested</th><th>Fitted values</th><th>Δ<Term k="bic" /></th><th><Term k="heldout">Held-out</Term> error</th></tr></thead>
            <tbody>
              {r.ranking.map((f) => (
                <tr key={f.model_id}>
                  <td data-label="Law">{f.title}{(f.latex ?? laws[f.model_id]?.latex) && <Law tex={(f.latex ?? laws[f.model_id]?.latex)!} />}<div className="muted small">{f.rationale ?? "Baseline, always tested as a control."}</div></td>
                  <td data-label="Fitted values" className="fitted">
                    {(f.param_meta ?? Object.keys(f.params).map((name) => ({ name, latex: name, unit: "" }))).map((p) => (
                      <div key={p.name}>
                        <Tex tex={p.latex} /> = {fmt(f.params[p.name])} {formatUnit(p.unit)}
                        {r.scale_note && needsScale(p.unit) && <span className="uncal">uncalibrated</span>}
                      </div>
                    ))}
                  </td>
                  <td data-label="ΔBIC" className="num">{f.bic - bestBic < 0.5 ? <StatusChip status="verified" label="best" /> : `+${(f.bic - bestBic).toFixed(0)}`}</td>
                  <td data-label="Held-out error" className="num">{(f.heldout_nrmse * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small" style={{ marginTop: 8 }}>
          ΔBIC is how much worse a model scores than the best one after a penalty for every extra parameter; above 10 is very
          strong evidence against it. Held-out error is how far its forecast of the last quarter, fitted without it, misses.
        </p>

      </div>
          </div>
        </section>
        <section id="exports" className="result-section" aria-labelledby="exports-h">
          <h2 id="exports-h">{SECTION_TITLE["exports"]}</h2>
          <div className="stack">
            {!r.fit_state && <p className="muted">No law was accepted, so there is nothing to export.</p>}
      {r.fit_state && (
        <div className="panel">
          <h3>Use the result</h3>
          <p className="muted small">
            Take the identified dynamics into a robot simulator or your own code.
            {r.mujoco?.rms_vs_fit_m != null && (r.mujoco.notes.length === 0 ? (
              <> MuJoCo replays the exported model within {mujocoGap(r.mujoco.rms_vs_fit_m, !!r.scale_note)} of the fitted motion (the green dot on the video).</>
            ) : (
              <> The MuJoCo model is an approximation (see the note below), so its replay departs from the fitted motion by{" "}
                {mujocoGap(r.mujoco.rms_vs_fit_m, !!r.scale_note)} RMS (the green dot); the Python simulation runs the exact equation.</>
            ))}
          </p>
          <div className="export-links">
            {r.mujoco && <a className="button" href={exportUrl(runId, "mjcf")} download><Boxes size={16} aria-hidden="true" />MuJoCo model</a>}
            <a className={`button ${r.mujoco ? "secondary" : ""}`} href={exportUrl(runId, "py")} download><FileCode2 size={16} aria-hidden="true" />Python simulation</a>
            <a className="button secondary" href={exportUrl(runId, "json")} download><FileJson size={16} aria-hidden="true" />Parameters</a>
            <a className="button secondary" href={exportUrl(runId, "report")} download><FileText size={16} aria-hidden="true" />Report</a>
            <button className="secondary" onClick={onShare} disabled={share.state === "busy"}>
              <Film size={16} aria-hidden="true" />{share.state === "busy" ? "Rendering video" : "Shareable video"}
            </button>
          </div>
          {r.mujoco?.notes.map((n, i) => <p key={i} className="muted small" style={{ marginTop: 8 }}>{n}</p>)}
          {!r.mujoco && (
            <p className="muted small" style={{ marginTop: 8 }}>
              No MuJoCo model for this law: MuJoCo has no built-in element for it. The Python simulation runs the exact fitted equation.
            </p>
          )}
          {share.state === "error" && <p className="error small" style={{ marginTop: 8 }}>{share.error}</p>}
          {share.state === "done" && share.url && (
            <>
              <video src={share.url} controls playsInline muted className="share-preview" />
              <p style={{ marginTop: 8 }}><a href={share.url} download><Download size={14} aria-hidden="true" /> Download the MP4</a></p>
            </>
          )}
          {(r.family === "pendulum" || r.family === "oscillator") && !isGallery(runId) && (
            <p className="small" style={{ marginTop: 12 }}><a href={`/health?baseline=${runId}`}>Compare this recording with a later one for wear</a></p>
          )}
        </div>
      )}
      {r.status === "accepted" && r.fit_state && online === true && (
        <ControlPanel runId={runId} family={r.family} params={(r.fit_state.params ?? {}) as Record<string, number>}
          length={(r.fit_state.ctx as Record<string, number> | undefined)?.length_m} />
      )}
      {r.fit_state && canAsk && <WhatIf runId={runId} onGhost={setWhatIfGhost} />}
          </div>
        </section>
        <section id="details" className="result-section" aria-labelledby="details-h">
          <h2 id="details-h">{SECTION_TITLE["details"]}</h2>
          <div className="stack">
            <div className="panel">
              <dl className="facts">
          <dt>Object</dt><dd>{r.scene.moving_object}. {r.scene.observations}</dd>
          <dt>Scale</dt><dd>{r.calibration.source === "none" ? "no reference in view (nominal scale)"
            : `${r.calibration.px_per_m.toFixed(1)} px per ${r.units?.length ?? "m"} from ${SCALE_SOURCE[r.calibration.source] ?? r.calibration.source.replaceAll("_", " ")}`}</dd>
          <dt>Clip</dt><dd>{r.clip.width} by {r.clip.height} px at {r.clip.fps.toFixed(1)} fps, {r.clip.frames} frames
            {r.clip.slow_motion && r.clip.slow_motion > 1 ? `, slow motion ${r.clip.slow_motion} times (times are real time)` : ""}
            {r.clip.slow_motion && r.clip.slow_motion < 1
              ? `, time-lapse: one frame is ${(1 / (r.clip.slow_motion * r.clip.fps)).toPrecision(3)} ${r.units?.time ?? "s"}` : ""}</dd>
          {r.sindy && (
            <><dt>Model-free regression</dt><dd>
              PySINDy, not verified and not used to decide: a sparse fit to noisy derivatives,
              which often disagrees with the verified equation in its small terms.
              <div className="sindy">{Object.entries(r.sindy).map(([lhs, rhs]) => <code key={lhs}>{lhs} = {rhs}</code>)}</div>
            </dd></>
          )}
          {r.full_clip_verdict && (
            <><dt>Whole clip</dt><dd>Fitted on the stretch the agent marked as free motion. Fitted on the whole clip instead, the verdict is
              {" "}<strong>{r.full_clip_verdict.status === "accepted" ? "accepted" : r.full_clip_verdict.status === "ambiguous" ? "a tie" : "no clean equation"}</strong>: {r.full_clip_verdict.reason}</dd></>
          )}
          {!r.sindy && r.sindy_check && !r.sindy_check.shown && r.sindy_check.reason !== "no verified law to compare with" && (
            <><dt>Model-free regression</dt><dd>PySINDy disagreed with the verified model, so it is hidden: {r.sindy_check.reason}.</dd></>
          )}
          <dt>Model calls</dt><dd>{r.llm_calls.map((c) => `${c.step}: ${c.model}${c.cached ? " (cached)" : ""}`).join(", ")}; tracking: sam-3.1</dd>
              </dl>
              {r.units && (
                <p className="small muted">
                  Units for this clip: lengths are in {r.units.length} and times in {r.units.time} (shown where the page says m and s).
                </p>
              )}
              {r.rolling_shutter?.note && <p className="note">{r.rolling_shutter.note}</p>}
              {r.camera_level?.note && <p className="note">{r.camera_level.note}</p>}
              {r.scale_basis && <p className="note">{r.scale_basis}</p>}
            </div>
      {r.audio && <AudioPanel audio={r.audio} cameraLawTitle={r.equation?.title} />}
      {r.depth3d && <DepthPanel d={r.depth3d} />}
          </div>
        </section>
    </section>
  );
}
