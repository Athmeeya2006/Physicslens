"use client";

import type { AnalysisResult } from "@/lib/api";
import { clipUrl } from "@/lib/api";
import { Tex } from "./Equation";
import { GhostPlayer } from "./GhostPlayer";

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
  return (
    <section className="result">
      <h2 className={`status ${r.status}`}>{STATUS_TEXT[r.status]}</h2>
      <p className="muted">{r.reason}</p>

      <GhostPlayer
        src={clipUrl(runId)}
        width={r.clip.width}
        height={r.clip.height}
        fps={r.clip.fps}
        real={r.real_track_px}
        ghosts={r.ghosts ?? []}
      />

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

      <div className="card agent">
        <h3>Agent</h3>
        <p>{r.agent.message}</p>
        <p className="muted"><strong>Reasoning:</strong> {r.agent.reasoning_summary}</p>
      </div>

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
