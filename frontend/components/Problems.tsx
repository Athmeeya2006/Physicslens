"use client";

import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

// The home page's "problems it solves": each real-world case as the problem someone has, the answer PhysicsLens gave
// and how that answer was checked. Read from the same generated file as /cases (scripts/build_cases.py), whose status
// rules are written down, so this page can never claim more than the results do. Proven cases come first.

interface Case { id: string; title: string; problem: string; who: string; number: string | null; status: string; link: string | null; models?: string }

const ORDER: Record<string, number> = { proven: 0, partial: 1, "not yet validated": 2 };

export function Problems() {
  const [cases, setCases] = useState<Case[]>([]);
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : null)).then((d) => d && setCases(d.cases)).catch(() => setCases([]));
  }, []);
  const shown = cases.filter((c) => c.number).sort((a, b) => (ORDER[a.status] ?? 3) - (ORDER[b.status] ?? 3));
  if (!shown.length) return null;
  const proven = shown.filter((c) => c.status === "proven").length;
  return (
    <section className="stack" style={{ gap: 14 }} aria-labelledby="problems-title">
      <div className="stack" style={{ gap: 4 }}>
        <h2 id="problems-title">Problems it solves</h2>
        <p className="muted">The same engine, on real data, each answer checked against an independent one: {proven} of {shown.length} proven so far,
          the rest labelled for what is still open.</p>
      </div>
      <ul className="problems">
        {shown.map((c) => (
          <li key={c.id} className="problem">
            <div className="problem-head">
              <h3>{c.title}</h3>
              <span className={`case-badge-inline case-${c.status.replaceAll(" ", "-")}`}>{c.status}</span>
            </div>
            <p className="problem-q">{c.problem}</p>
            <p className="problem-answer">{c.number}</p>
            {c.models && <p className="small muted">{c.models.charAt(0).toUpperCase() + c.models.slice(1)}.</p>}
            <div className="problem-foot">
              <span className="small muted">For {c.who}</span>
              {c.link && <a className="small" href={c.link}>See it <ArrowRight size={14} aria-hidden="true" /></a>}
            </div>
          </li>
        ))}
      </ul>
      <p><a className="button secondary" href="/cases">Every case, how it was checked, and what it can&apos;t do <ArrowRight size={16} aria-hidden="true" /></a></p>
    </section>
  );
}
