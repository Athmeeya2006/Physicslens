"use client";

import { Boxes, Download, FileCode2, FileJson, FileText, Film } from "lucide-react";
import { useState } from "react";
import type { AnalysisResult, TrackPoint } from "@/lib/api";
import { clipUrl, exportUrl, isGallery, isOpenWeight, makeShareVideo, useBackend, formatUnit } from "@/lib/api";
import { AgentTrail } from "./AgentTrail";
import { AudioPanel } from "./AudioPanel";
import { ControlPanel } from "./ControlPanel";
import { Tex } from "./Equation";
import { ForecastChart } from "./ForecastChart";
import { GhostPlayer } from "./GhostPlayer";
import { SimToReal } from "./SimToReal";
import { WhatIf } from "./WhatIf";

const STATUS_TEXT: Record<AnalysisResult["status"], string> = {
  accepted: "Verified equation",
  ambiguous: "Needs one more experiment",
  no_fit: "No clean governing equation",
};

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1000 || a < 0.01 ? v.toExponential(2) : v.toPrecision(4);
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

// Readable names for the tags; the exact model id stays in the tooltip for anyone verifying.
function modelName(id: string): string {
  if (id.startsWith("muse-glimmer")) return "Muse Glimmer 30B · open weights";
  if (id === "sam-3.1") return "SAM 3.1";
  if (id.startsWith("muse-spark")) return `Muse Spark 1.3${id.endsWith("contributor") ? " (development)" : ""}`;
  return id;
}

const SCALE_SOURCE: Record<string, string> = {
  user: "the length you gave", object_size: "the object's size you gave", gravity: "Earth's gravity (assumed)",
};

function prettyModel(id: string, titles: Record<string, string>): string {
  return titles[id] ?? id.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}

export function ResultView({ runId, result }: { runId: string; result: AnalysisResult }) {
  const r = result;
  const [whatIfGhost, setWhatIfGhost] = useState<{ label: string; points: TrackPoint[] } | null>(null);
  const [share, setShare] = useState<{ state: "idle" | "busy" | "done" | "error"; url?: string; error?: string }>({ state: "idle" });
  const titles = Object.fromEntries(r.ranking.map((f) => [f.model_id, f.title]));
  const models = Array.from(new Set((r.trail ?? []).map((t) => t.model).filter((m) => m && m !== "simulate-and-compare")));
  const online = useBackend();
  const canAsk = online === true || !isGallery(runId);
  const bestBic = Math.min(...r.ranking.map((f) => f.bic));

  const onShare = () => {
    setShare({ state: "busy" });
    makeShareVideo(runId)
      .then((url) => setShare({ state: "done", url }))
      .catch((e: Error) => setShare({ state: "error", error: e.message }));
  };

  return (
    <section className="stack" aria-labelledby="result-title">
      <div className="result-head">
        <h2 id="result-title" className={r.status}>{STATUS_TEXT[r.status]}</h2>
        <p className="reason">{r.reason}</p>
        <div className="provenance">
          <span>Models used:</span>
          {models.map((m) => (
            <span key={m} className={`model-tag ${isOpenWeight(m) ? "open" : ""}`} title={m}>{modelName(m)}</span>
          ))}
          <span className="model-tag">numerical verification</span>
        </div>
        {r.scale_note && <p className="note">{r.scale_note}</p>}
        {r.scale_basis && <p className="note">{r.scale_basis}</p>}
      </div>

      <div className="result-grid">
        <GhostPlayer
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
        />
        <div className="panel">
          <h3>Decision trail</h3>
          <AgentTrail steps={r.trail ?? []} runId={runId} />
        </div>
      </div>

      {r.forecast && (
        <div className="panel forecast">
          <h3>Forecast check: does the model predict motion it never saw?</h3>
          <ForecastChart f={r.forecast} uncalibrated={!!r.scale_note} />
        </div>
      )}

      {r.status === "accepted" && r.fit_state && <SimToReal runId={runId} />}

      {r.status === "accepted" && r.fit_state && online === true && (
        <ControlPanel runId={runId} family={r.family} params={(r.fit_state.params ?? {}) as Record<string, number>}
          length={(r.fit_state.ctx as Record<string, number> | undefined)?.length_m} />
      )}

      {r.equation && (
        <div className="panel">
          <h3>{r.equation.title}</h3>
          <div className="equation"><Tex tex={r.equation.latex} display /></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Parameter</th><th>Value</th><th className="ci-col">95% interval</th></tr></thead>
              <tbody>
                {r.parameters?.map((p) => {
                  const d = decimalsFor(p.std);
                  return (
                    <tr key={p.name}>
                      <td><Tex tex={p.latex} /> {p.derived && <span className="muted small">{p.derived}</span>}</td>
                      <td className="num"><span className="param-value">{fmtAt(p.value, d)}</span> ± {fmtAt(p.std, d)} {formatUnit(p.unit)}{" "}
                        {p.uncalibrated && <span className="uncal">uncalibrated</span>}
                        <div className="ci-inline muted small">95%: {fmtAt(p.ci95[0], d)} to {fmtAt(p.ci95[1], d)}</div></td>
                      <td className="num ci-col">{fmtAt(p.ci95[0], d)} to {fmtAt(p.ci95[1], d)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
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

      {r.audio && <AudioPanel audio={r.audio} cameraLawTitle={r.equation?.title} />}

      {r.fit_state && canAsk && <WhatIf runId={runId} onGhost={setWhatIfGhost} />}

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

      <div className="panel">
        <h3>Every candidate tested</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model, and why it was tested</th><th>Fitted values</th><th>ΔBIC</th><th>Held-out error</th></tr></thead>
            <tbody>
              {r.ranking.map((f) => (
                <tr key={f.model_id}>
                  <td>{f.title}<div className="muted small">{f.rationale ?? "Baseline, always tested as a control."}</div></td>
                  <td className="fitted">
                    {(f.param_meta ?? Object.keys(f.params).map((name) => ({ name, latex: name, unit: "" }))).map((p) => (
                      <div key={p.name}>
                        <Tex tex={p.latex} /> = {fmt(f.params[p.name])} {formatUnit(p.unit)}
                        {r.scale_note && needsScale(p.unit) && <span className="uncal">uncalibrated</span>}
                      </div>
                    ))}
                  </td>
                  <td className="num">{f.bic - bestBic < 0.5 ? "best" : `+${(f.bic - bestBic).toFixed(0)}`}</td>
                  <td className="num">{(f.heldout_nrmse * 100).toFixed(1)}%</td>
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

      <details className="panel">
        <summary>Measurement details</summary>
        <ul>
          <li>Object: {r.scene.moving_object}. {r.scene.observations}</li>
          <li>Scale: {r.calibration.source === "none" ? "no reference in view (nominal scale)"
            : `${r.calibration.px_per_m.toFixed(1)} px/m from ${SCALE_SOURCE[r.calibration.source] ?? r.calibration.source.replaceAll("_", " ")}`}</li>
          <li>Clip: {r.clip.width} by {r.clip.height} px at {r.clip.fps.toFixed(1)} fps, {r.clip.frames} frames
            {r.clip.slow_motion && r.clip.slow_motion > 1 ? `, slow motion ${r.clip.slow_motion} times (times are real time)` : ""}</li>
          {r.sindy && (
            <li>
              Model-free regression (PySINDy), not verified and not used to decide: a sparse fit to noisy derivatives,
              which often disagrees with the verified equation in its small terms.
              <div className="sindy">{Object.entries(r.sindy).map(([lhs, rhs]) => <code key={lhs}>{lhs} = {rhs}</code>)}</div>
            </li>
          )}
          {!r.sindy && r.sindy_check && !r.sindy_check.shown && r.sindy_check.reason !== "no verified law to compare with" && (
            <li>Model-free regression (PySINDy) disagreed with the verified model, so it is hidden: {r.sindy_check.reason}.</li>
          )}
          <li>Model calls: {r.llm_calls.map((c) => `${c.step}: ${c.model}${c.cached ? " (cached)" : ""}`).join(", ")}; tracking: sam-3.1</li>
        </ul>
      </details>
    </section>
  );
}
