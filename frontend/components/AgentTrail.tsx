"use client";

import {
  ClipboardList, Compass, Crosshair, Eye, Flag, FlaskConical, GitBranch, Layers, Lightbulb, MessageSquareText, Mic, Orbit, Ruler, ScanEye, Split,
  type LucideIcon,
} from "lucide-react";
import type { TrailStep } from "@/lib/api";

const STEP: Record<TrailStep["step"], { label: string; icon: LucideIcon }> = {
  plan: { label: "Plans the study", icon: ClipboardList },
  joint_fit: { label: "Fits the recordings together", icon: Layers },
  study_turn: { label: "Reviews the study", icon: Flag },
  scene: { label: "Looks at the scene", icon: Eye },
  tracking: { label: "Tracks the object", icon: Crosshair },
  track_check: { label: "Checks what it tracked", icon: ScanEye },
  hypotheses: { label: "Proposes equations", icon: Lightbulb },
  verification: { label: "Tests them against the data", icon: FlaskConical },
  design: { label: "Designs the next experiment", icon: Compass },
  plausibility: { label: "Checks the result is physically plausible", icon: Ruler },
  regime: { label: "Notices the dynamics changed", icon: Split },
  collision: { label: "Analyzes the bounces", icon: GitBranch },
  chaos: { label: "Measures chaos", icon: Orbit },
  audio: { label: "Listens to the soundtrack", icon: Mic },
  agent: { label: "Decides what to do next", icon: MessageSquareText },
};

const ACTION_LABEL: Record<string, string> = {
  explain: "Explains the verified result",
  request_experiment: "Asks for a new experiment",
  try_family: "Tries a different kind of model",
  give_up: "Reports no clean equation",
};

const ITEM_STATUS: Record<string, string> = { accepted: "Accepted", tied: "Tied", rejected: "Rejected", approximation: "Same law" };
// Candidate experiments are not models: say whether each would settle the question.
const DESIGN_STATUS: Record<string, string> = { accepted: "Decisive", tied: "Too close", rejected: "Too close" };

function actor(step: TrailStep): { name: string; tone: "open" | "meta" | "math" } {
  if (step.model === "simulate-and-compare") return { name: "Numerical simulation, not the language model", tone: "math" };
  if (step.model === "joint fit across recordings") return { name: "Numerical fitting across recordings, not the language model", tone: "math" };
  if (step.model === "acoustic analysis") return { name: "Signal processing on the microphone, not the language model", tone: "math" };
  if (step.model?.startsWith("muse-glimmer")) return { name: "Muse Glimmer 30B (open weights, self-hosted)", tone: "open" };
  if (step.model === "meta-models/Muse-Glimmer-30B") return { name: "Muse Glimmer 30B (open weights, hosted by Together AI)", tone: "open" };
  if (step.model === "sam-3.1") return { name: "SAM 3.1", tone: "meta" };
  // Only reachable in dev: demo and prod refuse to fall back from the open-weight model.
  return { name: `${step.model} (development fallback)`, tone: "meta" };
}

export function AgentTrail({ steps }: { steps: TrailStep[] }) {
  return (
    <ol className="trail">
      {steps.map((s, i) => {
        const who = actor(s);
        const highlight = s.step === "agent" && s.action === "request_experiment";
        const { label, icon: Icon } = STEP[s.step] ?? STEP.agent;
        return (
          <li key={i} className={`trail-step ${s.step} ${highlight ? "highlight" : ""}`}>
            <span className="trail-icon" aria-hidden="true"><Icon size={13} strokeWidth={2.2} /></span>
            <div>
              <div className="trail-head">
                <strong>{s.step === "agent" && s.action ? ACTION_LABEL[s.action] ?? label : label}</strong>
                <span className={`who ${who.tone}`}>{who.name}</span>
              </div>
              <p className={highlight ? "lead" : undefined}>{s.text}</p>
              {s.items && (
                <ul className="trail-items">
                  {s.items.map((it) => (
                    <li key={it.model_id} className={it.status ?? "proposed"}>
                      {it.status
                        ? <span className={`verdict ${it.status}`}>{(s.step === "design" ? DESIGN_STATUS : ITEM_STATUS)[it.status] ?? it.status}</span>
                        : <span className="verdict">Proposed</span>}
                      <span><span className="item-title">{it.title}</span> <span className="why">{it.why}</span></span>
                    </li>
                  ))}
                </ul>
              )}
              {s.reasoning_summary && <p className="small muted">Reasoning: {s.reasoning_summary}</p>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
