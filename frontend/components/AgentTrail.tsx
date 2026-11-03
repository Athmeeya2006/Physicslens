"use client";

import {
  Activity, ClipboardList, Code2, Compass, Cpu, Crosshair, Eye, Flag, FlaskConical, GitBranch, Layers, Lightbulb, MessageSquareText, Mic, Orbit,
  Link2, PencilRuler, Ruler, ScanEye, Split, Swords, TrendingUp, type LucideIcon,
} from "lucide-react";
import type { TrailStep } from "@/lib/api";
import { SetupGuide } from "./SetupGuide";
import { isGallery } from "@/lib/api";
import { useEffect, useState } from "react";
import { Law } from "./Equation";
import { StatusChip } from "./StatusChip";

// Every library law's equation by id (scripts/build_law_latex.py), fetched once: older results name laws by id only,
// and every law the agent touches is shown with its equation.
let lawsCache: Promise<Record<string, { latex: string }>> | null = null;
export function useLaws(): Record<string, { latex: string }> {
  const [laws, setLaws] = useState<Record<string, { latex: string }>>({});
  useEffect(() => {
    lawsCache ??= fetch("/laws.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    lawsCache.then(setLaws);
  }, []);
  return laws;
}

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
  identifiability: { label: "Finds parameters this clip cannot separate", icon: Link2 },
  inputs: { label: "Takes the known inputs", icon: Activity },
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
  residual_spectrum: "Looks for something else pulling",
};

const ITEM_STATUS: Record<string, string> = { accepted: "Accepted", tied: "Tied", rejected: "Rejected", approximation: "Same law" };
// The referee's challengers: "rejected" means the challenge failed and the accepted law stands.
const SKEPTIC_STATUS: Record<string, string> = {
  accepted: "Won", tied: "As good", rejected: "Refuted", approximation: "Same law", untestable: "Not testable",
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
  if (step.model?.startsWith("muse-spark")) return { name: `${step.model} (development fallback)`, tone: "meta" };
  // Anything else names a numerical method (for example the bootstrap refits behind a tie), not a model.
  return { name: `${step.model ? step.model.charAt(0).toUpperCase() + step.model.slice(1) : "Numerical method"}, not the language model`, tone: "math" };
}

// The written instruction of a designed experiment; the picture guide illustrates exactly this.
function designInstruction(step: TrailStep): string {
  const first = step.items?.[0]?.title;
  return first ? `${first}.` : step.text;
}

export function AgentTrail({ steps, runId, guideStored, latex = {} }:
  { steps: TrailStep[]; runId?: string; guideStored?: boolean; latex?: Record<string, string> }) {
  const laws = useLaws();
  const texFor = (id: string, own?: string | null) => own ?? latex[id] ?? laws[id]?.latex;
  return (
    <ol className="trail">
      {steps.map((s, i) => {
        const who = actor(s);
        const highlight = s.step === "agent" && s.action === "request_experiment";
        const tie = highlight || !!s.items?.some((it) => it.status === "tied");
        const { label, icon: Icon } = STEP[s.step] ?? STEP.agent;
        const extra = (
          <>
            {s.expectation && <p className="small"><span className="muted">Expects:</span> {s.expectation}</p>}
            {s.items && s.step === "design" && (
              <div className="table-wrap">
                <table className="trail-table">
                  <thead><tr><th>Experiment</th><th>Would it settle it?</th><th>Why</th></tr></thead>
                  <tbody>
                    {s.items.map((it) => (
                      <tr key={it.model_id}>
                        <td>{it.title}{texFor(it.model_id, it.latex) && <Law tex={texFor(it.model_id, it.latex)!} />}</td>
                        <td>{it.status ? <StatusChip status={it.status === "accepted" ? "accepted" : "tied"} label={DESIGN_STATUS[it.status] ?? it.status} />
                          : <StatusChip status="proposed" />}</td>
                        <td className="why">{it.why}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {s.items && s.step !== "design" && (
              <ul className="trail-items">
                {s.items.map((it) => (
                  <li key={it.model_id} className={it.status ?? "proposed"}>
                    <StatusChip status={it.status ?? "proposed"}
                      label={it.status ? (s.step === "skeptic" ? SKEPTIC_STATUS : ITEM_STATUS)[it.status] ?? it.status : "Proposed"} />
                    <span>
                      <span className="item-title">{it.title}</span> <span className="why">{it.why}</span>
                    </span>
                    {texFor(it.model_id, it.latex) ? <span className="item-law"><Law tex={texFor(it.model_id, it.latex)!} /></span> : <span />}
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
          </>
        );
        return (
          <li key={i} className={`trail-step ${s.step}${highlight ? " highlight" : ""}${tie ? " tie" : ""}`}>
            <span className="trail-icon" aria-hidden="true"><Icon size={13} strokeWidth={2.2} /></span>
            <div>
              <div className="trail-head">
                <strong>{s.label ?? (s.step === "agent" && s.action ? ACTION_LABEL[s.action] ?? label : label)}</strong>
                <span className={`who ${who.tone}`}>{who.name}</span>
              </div>
              <p className={highlight ? "lead" : undefined}>{s.text}</p>
              {extra}
              {runId && s.step === "design" && <SetupGuide runId={runId} stepRef={i} instruction={designInstruction(s)}
                canDraw={!isGallery(runId) || !!guideStored} />}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
