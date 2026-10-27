"use client";

import type { TrailStep } from "@/lib/api";

const STEP_LABEL: Record<TrailStep["step"], string> = {
  scene: "Looks at the scene",
  tracking: "Tracks the object",
  hypotheses: "Proposes equations",
  verification: "Tests them against the data",
  agent: "Decides what to do next",
};

const ACTION_LABEL: Record<string, string> = {
  explain: "Explains the verified result",
  request_experiment: "Asks for a new experiment",
  try_family: "Tries a different kind of model",
  give_up: "Reports no clean equation",
};

function actor(step: TrailStep): { name: string; tone: "open" | "meta" | "math" } {
  if (step.model === "simulate-and-compare") return { name: "Numerical simulation (code, not the LLM)", tone: "math" };
  if (step.model?.startsWith("meta-models/muse-spark-open")) return { name: "Muse Spark Open · open weights · Baseten", tone: "open" };
  if (step.model === "sam-3.1") return { name: "SAM 3.1 · Meta Model API", tone: "meta" };
  // Only reachable in dev: demo/prod refuse to fall back from the open-weight model.
  return { name: `${step.model} · Meta API (dev fallback)`, tone: "meta" };
}

export function AgentTrail({ steps }: { steps: TrailStep[] }) {
  return (
    <ol className="trail">
      {steps.map((s, i) => {
        const who = actor(s);
        const highlight = s.step === "agent" && s.action === "request_experiment";
        return (
          <li key={i} className={`trail-step ${s.step} ${highlight ? "highlight" : ""}`}>
            <div className="trail-head">
              <strong>{s.step === "agent" && s.action ? ACTION_LABEL[s.action] ?? STEP_LABEL.agent : STEP_LABEL[s.step]}</strong>
              <span className={`who ${who.tone}`}>{who.name}</span>
            </div>
            <p>{s.text}</p>
            {s.items && (
              <ul className="trail-items">
                {s.items.map((it) => (
                  <li key={it.model_id} className={it.status ?? "proposed"}>
                    {it.status && <span className={`chip ${it.status}`}>{it.status}</span>}
                    <span className="item-title">{it.title}</span>
                    <span className="muted">: {it.why}</span>
                  </li>
                ))}
              </ul>
            )}
            {s.reasoning_summary && <p className="muted small">Reasoning: {s.reasoning_summary}</p>}
          </li>
        );
      })}
    </ol>
  );
}
