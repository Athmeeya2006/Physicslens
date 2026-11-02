"use client";

import { useEffect, useMemo, useState } from "react";
import { ProofNav } from "@/components/ProofNav";

// One card per real-world case (scripts/build_cases.py): answer first (title, the problem in one line, the number
// against an independent truth, the status), evidence on demand (who, what they get, the models, the status rule and
// the ladder, the limits) in a disclosure. The status is the highest rung of the ladder the case has evidence for.

interface Rung { rung: string; evidence: string }
interface Case {
  id: string; title: string; problem: string; who: string; gives: string; number: string | null;
  status: string; checked?: string; ladder?: Rung[]; status_rule: string; cant: string; link: string | null; models?: string;
}

const LADDER = ["solved", "improves a public tool", "confirmed", "predicted", "proven", "partial", "not yet validated"];
const GROUP: Record<string, string> = {
  robotics: "Robotics", "robot-arm": "Robotics", lab: "Measurement", health: "Measurement", audit: "Video checks",
  timing: "Astronomy", fireball: "Astronomy", sun: "Astronomy", occultation: "Astronomy", asteroids: "Astronomy", satellites: "Astronomy",
};
const GROUPS = ["All", "Robotics", "Measurement", "Astronomy", "Video checks"];

function label(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function CasesPage() {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState("All");
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      // The strongest evidence leads: highest rung first.
      .then((d) => setCases([...d.cases].sort((a: Case, b: Case) => LADDER.indexOf(a.status) - LADDER.indexOf(b.status))))
      .catch((e) => setError(`The cases could not be loaded (${(e as Error).message}).`));
  }, []);
  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of cases ?? []) m.set(c.status, (m.get(c.status) ?? 0) + 1);
    return LADDER.filter((r) => m.has(r)).map((r) => `${m.get(r)} ${r}`).join(" · ");
  }, [cases]);
  const shown = (cases ?? []).filter((c) => group === "All" || GROUP[c.id] === group);

  return (
    <main id="main">
      <ProofNav />
      <header className="page-head">
        <h1>Real-world cases</h1>
        <p className="tagline">
          Each case ends in something a person can act on, is tested on real data and validated against an independent
          answer. Nothing is trained: each case is fitted from scratch. The status comes from each result file by a written rule.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {cases && (
        <>
          <p className="small"><strong>{cases.length} cases:</strong> {counts}</p>
          <div className="chips" role="group" aria-label="Filter cases">
            {GROUPS.map((g) => (
              <button key={g} type="button" className={`chip${group === g ? " on" : ""}`} aria-pressed={group === g} onClick={() => setGroup(g)}>{g}</button>
            ))}
          </div>
          <div className="case-grid">
            {shown.map((c) => (
              <article key={c.id} className={`panel case-card case-${c.status.replaceAll(" ", "-")}`} aria-labelledby={`case-${c.id}`}>
                <p className="case-badge" title={`Status rule: ${c.status_rule}`}>{label(c.status)}</p>
                <h2 id={`case-${c.id}`}>{c.title}</h2>
                <p className="small">{c.problem}</p>
                <p className="case-number">{c.number ?? "No result yet."}</p>
                {c.link && <a className="button secondary" href={c.link}>See the result</a>}
                <div className="case-details">
                  <p className="small"><strong>Who:</strong> {c.who}</p>
                  <p className="small"><strong>What they get:</strong> {c.gives}</p>
                  {c.models && <p className="small case-models"><strong>Meta models:</strong> {c.models}</p>}
                  <p className="small"><strong>Status rule:</strong> {c.status_rule}</p>
                  {c.ladder && c.ladder.length > 1 && (
                    <ul className="small">{c.ladder.map((r) => <li key={r.rung}><strong>{label(r.rung)}:</strong> {r.evidence}</li>)}</ul>
                  )}
                  <p className="small muted"><strong>What it can&apos;t do:</strong> {c.cant}</p>
                </div>
              </article>
            ))}
          </div>
          <p className="small muted">The ladder, lowest to highest: not yet validated, partial, proven, predicted (a blind prediction
            registered before the answer existed), confirmed (reality scored it a hit), improves a public tool, solved (someone outside
            the project used it; recorded, never inferred). Open predictions are on the <a href="/predictions">predictions page</a>.</p>
        </>
      )}
    </main>
  );
}
