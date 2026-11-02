"use client";

import { useEffect, useState } from "react";
import { ProofNav } from "@/components/ProofNav";

// One card per real-world case (scripts/build_cases.py): the problem, who has it, what PhysicsLens gives them, one
// number against an independent truth, the status from the result file, and what it cannot do.

interface Case {
  id: string; title: string; problem: string; who: string; gives: string; number: string | null;
  status: "proven" | "partial" | "not yet validated"; status_rule: string; cant: string; link: string | null; models?: string;
}

const BADGE: Record<Case["status"], string> = { proven: "Proven", partial: "Partial", "not yet validated": "Not yet validated" };

const RANK: Record<string, number> = { proven: 0, partial: 1, "not yet validated": 2 };

export default function CasesPage() {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      // Proven first, then partial, then what is not validated yet: the strongest evidence leads.
      .then((d) => setCases([...d.cases].sort((a: Case, b: Case) => (RANK[a.status] ?? 3) - (RANK[b.status] ?? 3)))).catch((e) => setError(`The cases could not be loaded (${(e as Error).message}).`));
  }, []);

  return (
    <main id="main">
      <ProofNav />
      <header className="page-head">
        <h1>Real-world cases</h1>
        <p className="tagline">
          PhysicsLens finds the law, and the law tells you what to do. Each case ends in something a person can act on, is
          tested on real data and validated against an independent answer, and says what it cannot do. The status comes
          from each result file by the rule shown, never set by hand.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {cases && (
        <div className="case-grid">
          {cases.map((c) => (
            <article key={c.id} className={`panel case-card case-${c.status.replaceAll(" ", "-")}`} aria-labelledby={`case-${c.id}`}>
              <p className="case-badge" title={`Status rule: ${c.status_rule}`}>{BADGE[c.status]}</p>
              <h2 id={`case-${c.id}`}>{c.title}</h2>
              <p>{c.problem}</p>
              <p className="small"><strong>Who:</strong> {c.who}</p>
              <p className="small"><strong>What they get:</strong> {c.gives}</p>
              <p className="case-number">{c.number ?? "No result yet."}</p>
              {c.models && <p className="small case-models"><strong>Meta models:</strong> {c.models}</p>}
              <p className="small muted"><strong>What it can&apos;t do:</strong> {c.cant}</p>
              {c.link && <a className="button secondary" href={c.link}>See the result</a>}
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
