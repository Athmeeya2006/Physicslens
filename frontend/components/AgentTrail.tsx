"use client";

import {
  ClipboardList, Code2, Compass, Cpu, Crosshair, Eye, Flag, FlaskConical, GitBranch, Layers, Lightbulb, MessageSquareText, Mic, Orbit,
  PencilRuler, Ruler, ScanEye, Split, Swords, TrendingUp, type LucideIcon,
} from "lucide-react";
import type { TrailStep } from "@/lib/api";
import { SetupGuide } from "./SetupGuide";

const STEP: Record<TrailStep["step"], { label: string; icon: LucideIcon }> = {
  plan: { label: "Plans the study", icon: ClipboardList },
  joint_fit: { label: "Fits the recordings together", icon: Layers },
  study_turn: { label: "Reviews the study", icon: Flag },
  scene: { label: "Looks at the scene", icon: Eye },
  tracking: { label: "Tracks the object", icon: Crosshair },
  track_check: { label: "Checks what it tracked", icon: ScanEye },
  hypotheses: { label: "Proposes equations", icon: Lightbulb },
  verification: { label: "Tests them against the data", icon: FlaskConical },
  skeptic: { label: "Tries to break the accepted law", icon: Swords },
  scaling: { label: "Fits a law across recordings", icon: TrendingUp },
  locate: { label: "Locates the object precisely", icon: Crosshair },
  design: { label: "Designs the next experiment", icon: Compass },
  plausibility: { label: "Checks the result is physically plausible", icon: Ruler },
  regime: { label: "Notices the dynamics changed", icon: Split },
  collision: { label: "Analyzes the bounces", icon: GitBranch },
  chaos: { label: "Measures chaos", icon: Orbit },
  audio: { label: "Listens to the soundtrack", icon: Mic },
  agent: { label: "Decides what to do next", icon: MessageSquareText },
  setup_guide: { label: "Draws the setup", icon: PencilRuler },
  controller: { label: "Writes the controller", icon: Code2 },
  control_test: { label: "Tests it in simulation", icon: Cpu },
};

const ACTION_LABEL: Record<string, string> = {
  explain: "Explains the verified result",
  request_experiment: "Asks for a new experiment",
  try_family: "Tries a different kind of model",
  give_up: "Reports no clean equation",
  test_models: "Tests models of its own choosing",
  inspect: "Inspects part of the motion",
};

const ITEM_STATUS: Record<string, string> = { accepted: "Accepted", tied: "Tied", rejected: "Rejected", approximation: "Same law" };
// The referee's challengers: "rejected" means the challenge failed and the accepted law stands.
const SKEPTIC_STATUS: Record<string, string> = {
  accepted: "Won", tied: "As good", rejected: "Failed", approximation: "Same law", untestable: "Not testable",
};
// Candidate experiments are not models: say whether each would settle the question.
const DESIGN_STATUS: Record<string, string> = { accepted: "Decisive", tied: "Too close", rejected: "Too close" };

function actor(step: TrailStep): { name: string; tone: "open" | "meta" | "math" } {
  if (step.model === "simulate-and-compare") return { name: "Numerical simulation, not the language model", tone: "math" };
  if (step.model === "joint fit across recordings") return { name: "Numerical fitting across recordings, not the language model", tone: "math" };
  if (step.model === "MuJoCo simulation") return { name: "MuJoCo simulation, not the language model", tone: "math" };
  if (step.model === "muse-wonder-1.0") return { name: "Muse Wonder (Meta Model API)", tone: "meta" };
  if (step.model === "acoustic analysis") return { name: "Signal processing on the microphone, not the language model", tone: "math" };
  if (step.model?.startsWith("meta-models/muse-spark-open")) return { name: "Muse Spark Open (open weights, served by Baseten)", tone: "open" };
  if (step.model?.startsWith("muse-glimmer")) return { name: "Muse Glimmer 30B (open weights, self-hosted)", tone: "open" };
  if (step.model === "meta-models/Muse-Glimmer-30B") return { name: "Muse Glimmer 30B (open weights, hosted by Together AI)", tone: "open" };
  if (step.model === "sam-3.1") return { name: "SAM 3.1", tone: "meta" };
  if (step.model === "muse-image-1.0") return { name: "Muse Image (draws only, never measures)", tone: "meta" };
  // Only reachable in dev: demo and prod refuse to fall back from the open-weight model.
  return { name: `${step.model} (development fallback)`, tone: "meta" };
}

// The written instruction of a designed experiment; the picture guide illustrates exactly this.
function designInstruction(step: TrailStep): string {
  const first = step.items?.[0]?.title;
  return first ? `${first}.` : step.text;
}

export function AgentTrail({ steps, runId }: { steps: TrailStep[]; runId?: string }) {
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
                <strong>{s.label ?? (s.step === "agent" && s.action ? ACTION_LABEL[s.action] ?? label : label)}</strong>
                <span className={`who ${who.tone}`}>{who.name}</span>
              </div>
              {s.expectation && <p className="small"><span className="muted">Expects:</span> {s.expectation}</p>}
              <p className={highlight ? "lead" : undefined}>{s.text}</p>
              {s.items && (
                <ul className="trail-items">
                  {s.items.map((it) => (
                    <li key={it.model_id} className={it.status ?? "proposed"}>
                      {it.status
                        ? <span className={`verdict ${it.status}`}>{(s.step === "design" ? DESIGN_STATUS : s.step === "skeptic" ? SKEPTIC_STATUS : ITEM_STATUS)[it.status] ?? it.status}</span>
                        : <span className="verdict">Proposed</span>}
                      <span><span className="item-title">{it.title}</span> <span className="why">{it.why}</span></span>
                    </li>
                  ))}
                </ul>
              )}
              {s.reasoning_summary && <p className="small muted">Reasoning: {s.reasoning_summary}</p>}
              {s.fallbacks?.map((f, j) => <p key={j} className="small muted">Switched model: {f}</p>)}
              {s.reasoning && (
                <details className="thought">
                  <summary>What the model thought</summary>
                  <p>{s.reasoning}</p>
                </details>
              )}
              {runId && s.step === "design" && <SetupGuide runId={runId} stepRef={i} instruction={designInstruction(s)} />}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
