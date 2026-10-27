"use client";

import { useState } from "react";
import type { AnalysisResult } from "@/lib/api";
import { clipUrl, exportUrl, makeShareVideo } from "@/lib/api";
import { AgentTrail } from "./AgentTrail";
import { Tex } from "./Equation";
import { GhostPlayer } from "./GhostPlayer";
import { WhatIf } from "./WhatIf";
import type { TrackPoint } from "@/lib/api";

const STATUS_TEXT: Record<AnalysisResult["status"], string> = {
  accepted: "Verified equation",
  ambiguous: "More data needed",
  no_fit: "No clean governing equation found",
};

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1000 || a < 0.01 ? v.toExponential(2) : v.toPrecision(4);
}

export function ResultView({ runId, result }: { runId: string; result: AnalysisResult }) {
  const r = result;
  const [whatIfGhost, setWhatIfGhost] = useState<{ label: string; points: TrackPoint[] } | null>(null);
  const [share, setShare] = useState<{ state: "idle" | "busy" | "done" | "error"; url?: string; error?: string }>({ state: "idle" });

  const onShare = () => {
    setShare({ state: "busy" });
    makeShareVideo(runId)
      .then((url) => setShare({ state: "done", url }))
      .catch((e: Error) => setShare({ state: "error", error: e.message }));
  };

  return (
    <section className="result">
      <h2 className={`status ${r.status}`}>{STATUS_TEXT[r.status]}</h2>
      <p className="muted">{r.reason}</p>

      <div className="split">
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
          ghosts={r.ghosts ?? []}
        />
        <div className="card trail-card">
          <h3>Agent decision trail</h3>
          <AgentTrail steps={r.trail ?? []} />
        </div>
      </div>

      {r.fit_state && <WhatIf runId={runId} onGhost={setWhatIfGhost} />}

      {r.fit_state && (
        <div className="card exports">
          <h3>Export for simulation</h3>
          <p className="muted small">
            Use the identified dynamics in a robot simulator or your own code.
            {r.mujoco?.rms_vs_fit_m != null && (
              <> MuJoCo replays the exported model within {(r.mujoco.rms_vs_fit_m * 1000).toFixed(r.mujoco.rms_vs_fit_m < 0.001 ? 3 : 1)} mm of the fitted motion (green dot on the video).</>
            )}
          </p>
          <div className="export-links">
            <a href={exportUrl(runId, "mjcf")} download>MuJoCo model (MJCF)</a>
            <a href={exportUrl(runId, "py")} download>Python simulation</a>
            <a href={exportUrl(runId, "json")} download>Parameters (JSON)</a>
          </div>
          {r.mujoco?.notes.map((n, i) => <p key={i} className="muted small">Note: {n}</p>)}
          {(r.family === "pendulum" || r.family === "oscillator") && (
            <p className="small"><a href={`/health?baseline=${runId}`}>Machine health: compare this recording with a later one</a></p>
          )}
        </div>
      )}

      <div className="card share">
        <button onClick={onShare} disabled={share.state === "busy"}>
          {share.state === "busy" ? "Rendering video..." : "Make shareable video"}
        </button>
        {share.state === "error" && <p className="error">{share.error}</p>}
        {share.state === "done" && share.url && (
          <>
            <video src={share.url} controls playsInline muted className="share-preview" />
            <a href={share.url} download>Download MP4</a>
          </>
        )}
      </div>

      {r.double && (
        <div className="card">
          <h3>Double pendulum: {r.double.chaotic ? "chaotic" : "regular"} motion</h3>
          <div className="equation"><Tex tex={r.double.latex} display /></div>
          <p>
            <Tex tex="g" /> = {r.double.params.g.toFixed(3)} m/s², mass ratio <Tex tex="\mu" /> = {r.double.params.mu.toFixed(2)},{" "}
            <Tex tex="L_1" /> = {r.double.L1.toFixed(3)} m, <Tex tex="L_2" /> = {r.double.L2.toFixed(3)} m
          </p>
          <p>
            {r.double.chaotic && r.double.predictability_s
              ? `Nearby futures separate e-fold every ${(1 / r.double.lyapunov_per_s).toFixed(2)} s (Lyapunov exponent ${r.double.lyapunov_per_s.toFixed(1)}/s). With this tracking noise the motion can be forecast for about ${r.double.predictability_s.toFixed(1)} s, no further.`
              : `Lyapunov exponent ${r.double.lyapunov_per_s.toFixed(2)}/s: small differences stay small, so the motion stays predictable.`}
          </p>
          <p className="muted small">
            Fitted on short predictions restarted from the measured state, because whole-clip simulations of a chaotic system always drift.
            The yellow ghost is one prediction from the start, to show that drift.
          </p>
        </div>
      )}

      {r.bounces && (
        <div className="card">
          <h3>Bounces: {r.bounces.accepted ? "hybrid model verified" : "hybrid model not confirmed"}</h3>
          <div className="equation"><Tex tex={r.bounces.latex} display /></div>
          <p>
            <Tex tex="g" /> = {r.bounces.g.toFixed(3)} ± {r.bounces.g_std.toFixed(3)} m/s², coefficient of restitution{" "}
            <Tex tex="e" /> = {r.bounces.e_mean.toFixed(3)} ± {r.bounces.e_std.toFixed(3)}
          </p>
          <table>
            <thead><tr><th>Impact</th><th>Time</th><th>e</th><th>Horizontal speed kept</th></tr></thead>
            <tbody>
              {r.bounces.impacts.map((imp, i) => (
                <tr key={i}>
                  <td>{i + 1}</td><td>{imp.t.toFixed(2)} s</td>
                  <td>{imp.e.toFixed(3)} ± {imp.e_std.toFixed(3)}</td><td>{(imp.tangential * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            One g is shared by every flight. Later bounces predicted from the first ones within {(r.bounces.forecast_nrmse * 100).toFixed(1)}%.
            {r.bounces.settled_at ? ` After ${r.bounces.settled_at.toFixed(1)} s the hops were too small to resolve.` : ""}
          </p>
        </div>
      )}

      {r.equation && (
        <div className="card">
          <h3>{r.equation.title}</h3>
          <div className="equation">
            <Tex tex={r.equation.latex} display />
          </div>
          <table>
            <thead>
              <tr><th>Parameter</th><th>Value</th><th>95% interval</th></tr>
            </thead>
            <tbody>
              {r.parameters?.map((p) => (
                <tr key={p.name}>
                  <td><Tex tex={p.latex} /> {p.derived && <span className="muted">({p.derived})</span>}</td>
                  <td>{fmt(p.value)} ± {fmt(p.std)} {p.unit}</td>
                  <td>[{fmt(p.ci95[0])}, {fmt(p.ci95[1])}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h3>Candidates tested</h3>
        <table>
          <thead>
            <tr><th>Model</th><th>BIC</th><th>Forecast error</th><th>Why proposed</th></tr>
          </thead>
          <tbody>
            {r.ranking.map((f) => (
              <tr key={f.model_id}>
                <td>{f.title}</td>
                <td>{f.bic.toFixed(0)}</td>
                <td>{(f.heldout_nrmse * 100).toFixed(1)}%</td>
                <td className="muted">{f.rationale ?? "baseline control"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {r.sindy && (
          <p className="muted">
            Model-free discovery (PySINDy):{" "}
            {Object.entries(r.sindy).map(([lhs, rhs]) => (
              <code key={lhs}>{lhs} = {rhs}</code>
            ))}
          </p>
        )}
      </div>

      <details className="card">
        <summary>Measurement details</summary>
        <ul>
          <li>Object: {r.scene.moving_object} ({r.scene.observations})</li>
          <li>Scale: {r.calibration.px_per_m.toFixed(1)} px/m from {r.calibration.source.replaceAll("_", " ")}</li>
          <li>Clip: {r.clip.width}×{r.clip.height} at {r.clip.fps.toFixed(1)} fps, {r.clip.frames} frames</li>
          <li>Models used: {r.llm_calls.map((c) => `${c.step}: ${c.model}${c.cached ? " (cached)" : ""}`).join(", ")}, tracking: sam-3.1</li>
        </ul>
      </details>
    </section>
  );
}
