"use client";

import { useEffect, useState } from "react";
import type { TrailStep } from "@/lib/api";
import { AgentTrail } from "./AgentTrail";
import { Law } from "./Equation";

// A real-world case run through the agent (scripts/cases_agent.py): Muse Spark Open saw only the numbers and proposed
// its own laws, verification ranked them, and the hand-chosen laws sit beside them. Shown only when the run exists.

interface Row { model_id: string; title: string; heldout_nrmse: number | null; status?: string; why?: string }
interface AgentCase {
  status: string; reason: string; equation?: { title: string; latex: string } | null; agent_ranking: Row[];
  agent_proposed_free_form: string[]; hand_chosen: { law: string; title: string }[]; trail?: TrailStep[];
  llm_calls?: { provider: string; model: string }[]; mock?: boolean;
}

export function AgentOnCase({ file, what }: { file: string; what: string }) {
  const [d, setD] = useState<AgentCase | null>(null);
  useEffect(() => {
    fetch(`/astro/${file}`).then((r) => (r.ok ? r.json() : null)).then((x) => setD(x && !x.mock ? x : null)).catch(() => setD(null));
  }, [file]);
  if (!d) return null;
  const model = d.llm_calls?.[0]?.model ?? "Muse Spark Open";
  return (
    <div className="panel stack agent-case">
      <h3>The agent on the same data</h3>
      <p className="small">
        {model} (Meta&apos;s open-weight model) saw only {what} and proposed its own laws; the same held-out verification as for
        a video ranked them. Verdict: <strong>{d.status}</strong>. {d.reason}
      </p>
      {d.equation && <Law tex={d.equation.latex} />}
      {d.agent_proposed_free_form.length > 0 && (
        <p className="small"><strong>Laws the agent wrote itself:</strong> {d.agent_proposed_free_form.join("; ")}.</p>
      )}
      <div className="table-wrap">
        <table className="cards">
          <thead><tr><th>Law proposed by the agent</th><th>Verdict</th><th>Forecast error on held-out data</th></tr></thead>
          <tbody>
            {d.agent_ranking.map((r) => (
              <tr key={r.model_id}>
                <td data-label="Law">{r.title}</td>
                <td data-label="Verdict">{r.status || "tested"}{r.why ? `: ${r.why}` : ""}</td>
                <td data-label="Held-out" className="num">{r.heldout_nrmse != null ? `${(r.heldout_nrmse * 100).toFixed(1)}%` : "n/a"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Laws a person chose for this case, shown above: {d.hand_chosen.map((h) => h.title).join("; ")}.</p>
      {d.trail && <><h4>Its decision trail</h4><AgentTrail steps={d.trail} /></>}
    </div>
  );
}
