"use client";

import { Download, Film, FileCode2, FileJson, Boxes } from "lucide-react";
import { useState } from "react";
import type { AnalysisResult, TrackPoint } from "@/lib/api";
import { clipUrl, exportUrl, HAS_BACKEND, isGallery, makeShareVideo } from "@/lib/api";
import { AgentTrail } from "./AgentTrail";
import { Tex } from "./Equation";
import { ForecastChart } from "./ForecastChart";
import { GhostPlayer } from "./GhostPlayer";
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

function prettyModel(id: string, titles: Record<string, string>): string {
  return titles[id] ?? id.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}

export function ResultView({ runId, result }: { runId: string; result: AnalysisResult }) {
  const r = result;
  const [whatIfGhost, setWhatIfGhost] = useState<{ label: string; points: TrackPoint[] } | null>(null);
  const [share, setShare] = useState<{ state: "idle" | "busy" | "done" | "error"; url?: string; error?: string }>({ state: "idle" });
  const titles = Object.fromEntries(r.ranking.map((f) => [f.model_id, f.title]));
  const models = Array.from(new Set((r.trail ?? []).map((t) => t.model).filter((m) => m && m !== "simulate-and-compare")));
  const canAsk = HAS_BACKEND || !isGallery(runId);

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
            <span key={m} className={`model-tag ${m.startsWith("meta-models/muse-spark-open") ? "open" : ""}`}>{m}</span>
          ))}
          <span className="model-tag">numerical verification</span>
        </div>
        {r.scale_note && <p className="note">{r.scale_note}</p>}
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
          <AgentTrail steps={r.trail ?? []} />
        </div>
      </div>

      {r.forecast && (
        <div className="panel forecast">
          <h3>Forecast check: does the model predict motion it never saw?</h3>
          <ForecastChart f={r.forecast} uncalibrated={!!r.scale_note} />
        </div>
      )}

      {r.equation && (
        <div className="panel">
          <h3>{r.equation.title}</h3>
          <div className="equation"><Tex tex={r.equation.latex} display /></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Parameter</th><th>Value</th><th>95% interval</th></tr></thead>
              <tbody>
                {r.parameters?.map((p) => (
                  <tr key={p.name}>
                    <td><Tex tex={p.latex} /> {p.derived && <span className="muted small">{p.derived}</span>}</td>
                    <td className="num"><span className="param-value">{fmt(p.value)}</span> ± {fmt(p.std)} {p.unit}{" "}
                      {p.uncalibrated && <span className="uncal">uncalibrated</span>}</td>
                    <td className="num">{fmt(p.ci95[0])} to {fmt(p.ci95[1])}</td>
                  </tr>
                ))}
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
            <Tex tex="g" /> = {r.bounces.g.toFixed(3)} ± {r.bounces.g_std.toFixed(3)} m/s², coefficient of restitution{" "}
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
            One g is shared by every flight. Later bounces are predicted from the first ones within {(r.bounces.forecast_nrmse * 100).toFixed(1)}%.
            {r.bounces.settled_at ? ` After ${r.bounces.settled_at.toFixed(1)} s the hops were too small to resolve.` : ""}
          </p>
        </div>
      )}

      {r.fit_state && canAsk && <WhatIf runId={runId} onGhost={setWhatIfGhost} />}

      {r.fit_state && (
        <div className="panel">
          <h3>Use the result</h3>
          <p className="muted small">
            Take the identified dynamics into a robot simulator or your own code.
            {r.mujoco?.rms_vs_fit_m != null && (
              <> MuJoCo replays the exported model within {(r.mujoco.rms_vs_fit_m * 1000).toFixed(r.mujoco.rms_vs_fit_m < 0.001 ? 3 : 1)} mm of the fitted motion (the green dot on the video).</>
            )}
          </p>
          <div className="export-links">
            <a className="button" href={exportUrl(runId, "mjcf")} download><Boxes size={16} aria-hidden="true" />MuJoCo model</a>
            <a className="button secondary" href={exportUrl(runId, "py")} download><FileCode2 size={16} aria-hidden="true" />Python simulation</a>
            <a className="button secondary" href={exportUrl(runId, "json")} download><FileJson size={16} aria-hidden="true" />Parameters</a>
            <button className="secondary" onClick={onShare} disabled={share.state === "busy"}>
              <Film size={16} aria-hidden="true" />{share.state === "busy" ? "Rendering video" : "Shareable video"}
            </button>
          </div>
          {r.mujoco?.notes.map((n, i) => <p key={i} className="muted small" style={{ marginTop: 8 }}>{n}</p>)}
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
            <thead><tr><th>Model</th><th>BIC</th><th>Held-out forecast error</th><th>Why it was proposed</th></tr></thead>
            <tbody>
              {r.ranking.map((f) => (
                <tr key={f.model_id}>
                  <td>{f.title}</td>
                  <td className="num">{f.bic.toFixed(0)}</td>
                  <td className="num">{(f.heldout_nrmse * 100).toFixed(1)}%</td>
                  <td className="muted">{f.rationale ?? "baseline, always tested as a control"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {r.sindy && (
          <div className="sindy" style={{ marginTop: 12 }}>
            <p className="small muted">Model-free discovery with PySINDy, shown as independent evidence:</p>
            {Object.entries(r.sindy).map(([lhs, rhs]) => <code key={lhs}>{lhs} = {rhs}</code>)}
          </div>
        )}
      </div>

      <details className="panel">
        <summary>Measurement details</summary>
        <ul>
          <li>Object: {r.scene.moving_object}. {r.scene.observations}</li>
          <li>Scale: {r.calibration.source === "none" ? "no reference in view (nominal scale)" : `${r.calibration.px_per_m.toFixed(1)} px/m from ${r.calibration.source.replaceAll("_", " ")}`}</li>
          <li>Clip: {r.clip.width} by {r.clip.height} px at {r.clip.fps.toFixed(1)} fps, {r.clip.frames} frames</li>
          <li>Model calls: {r.llm_calls.map((c) => `${c.step}: ${c.model}${c.cached ? " (cached)" : ""}`).join(", ")}; tracking: sam-3.1</li>
        </ul>
      </details>
    </section>
  );
}
