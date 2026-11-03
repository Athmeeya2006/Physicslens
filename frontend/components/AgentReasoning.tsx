import type { TrailStep } from "@/lib/api";

// How the agent reasoned, in a few lines at the top of a result: the decisive moments of the decision trail, in order,
// each with who did it (a Meta model or the math) and, where the model gave one, its own reasoning summary. Built from
// the stored trail only: nothing here is newly written by a model, and the full trail follows below.

const KEY = new Set(["scene", "track_check", "hypotheses", "verification", "skeptic", "design", "agent", "agreed", "regime", "collision"]);

function who(model?: string | null): { name: string; math: boolean } {
  if (!model || model === "simulate-and-compare" || !/^(meta-models\/|muse-|sam-)/.test(model)) return { name: "the math", math: true };
  if (model.startsWith("meta-models/muse-spark-open")) return { name: "Muse Spark Open", math: false };
  if (model.startsWith("muse-glimmer") || model === "meta-models/Muse-Glimmer-30B") return { name: "Muse Glimmer 30B", math: false };
  if (model === "sam-3.1") return { name: "SAM 3.1", math: false };
  return { name: model, math: false };
}

const first = (t: string) => {
  const m = t.match(/^(.+?[.!?])(\s|$)/);
  return m ? m[1] : t;
};

function line(s: TrailStep): { verb: string; text: string; tone?: "reject" | "tie" | "ask" } | null {
  const items = s.items ?? [];
  if (s.step === "scene") return { verb: "Sees", text: first(s.text).replace(/^Sees /, "") };
  if (s.step === "track_check") return /not the|asks SAM again|re-track/i.test(s.text) ? { verb: "Catches a bad track", text: first(s.text) } : null;
  if (s.step === "hypotheses") {
    const names = items.map((i) => i.title).filter(Boolean);
    return { verb: "Proposes", text: names.length ? `${names.join(", ")}.` : first(s.text) };
  }
  if (s.step === "verification") {
    const rej = items.filter((i) => i.status === "rejected");
    const tied = items.filter((i) => i.status === "tied");
    const acc = items.filter((i) => i.status === "accepted");
    const rejected = rej.slice(0, 3).map((i) => `${i.title} (${i.why})`).join("; ");
    // A tie is the point of the step when there is one: say it first, then what was rejected on the way.
    if (tied.length) return { verb: "Cannot separate", tone: "tie",
      text: `${tied.map((i) => i.title).join(" and ")} forecast the unseen motion about equally well.` + (rej.length ? ` Rejects ${rejected}.` : "") };
    if (rej.length) return { verb: "Rejects", tone: "reject", text: rejected + (acc.length ? `. Keeps ${acc[0].title}.` : ".") };
    return { verb: "Verifies", text: first(s.text) };
  }
  if (s.step === "skeptic") {
    // Only the referee's decisive moments: it changed the verdict, or the law survived its challenges.
    if (/verdict changes/i.test(s.text)) return { verb: "Overturns the verdict", tone: "reject", text: first(s.text) };
    if (/survived/i.test(s.text)) return { verb: "Tries to break it", text: first(s.text) };
    return null;
  }
  if (s.step === "design") return { verb: "Designs the deciding experiment", text: first(s.text.replace(/^Simulated [^.]*\.\s*/, "")) };
  if (s.step === "agent" && s.action === "request_experiment") return { verb: "Asks you for", tone: "ask", text: first(s.text) };
  if (s.step === "agent") return { verb: "Decides", text: first(s.text) };
  if ((s.step as string) === "agreed") return { verb: "Still measures", text: first(s.text) };
  if (s.step === "regime") return { verb: "Notices", text: first(s.text) };
  if (s.step === "collision") return { verb: "Finds", text: first(s.text) };
  return null;
}

export function AgentReasoning({ steps }: { steps: TrailStep[] }) {
  const rows = steps.filter((s) => KEY.has(s.step as string)).map((s) => ({ s, l: line(s) })).filter((r) => r.l);
  if (!rows.length) return null;
  return (
    <section className="agent-reasoning" aria-labelledby="ar-title">
      <h2 id="ar-title">How the agent reasoned</h2>
      <ol>
        {rows.map(({ s, l }, i) => {
          const w = who(s.model);
          return (
            <li key={i} className={l!.tone ?? ""}>
              <span className="ar-verb">{l!.verb}</span>
              <span className="ar-text">{l!.text}
                {s.reasoning_summary && !w.math && <q className="ar-quote">{first(s.reasoning_summary)}</q>}
              </span>
              <span className={`ar-who${w.math ? " math" : ""}`}>{w.name}</span>
            </li>
          );
        })}
      </ol>
      <p className="small">The language model proposes and plans; only the math accepts or rejects a law. Every step, with the model&apos;s
        full reasoning, is in the <a href="#trail">decision trail</a>.</p>
    </section>
  );
}
