"use client";

import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

// The home page's view of the real-world cases (/cases): each case's status and its one number, from the same
// generated file, so the home page can never claim more than the results do.

interface Case { id: string; title: string; number: string | null; status: string; link: string | null }

export function CasesStrip() {
  const [cases, setCases] = useState<Case[]>([]);
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : null)).then((d) => d && setCases(d.cases)).catch(() => setCases([]));
  }, []);
  const shown = cases.filter((c) => c.number);
  if (!shown.length) return null;
  return (
    <section className="panel stack cases-strip" aria-labelledby="cases-strip-title">
      <h2 id="cases-strip-title">Real-world cases</h2>
      <p className="small muted">The law is the engine; what someone uses is what it tells them to do. Each case is tested on real data
        and validated against an independent answer.</p>
      <ul>
        {shown.map((c) => (
          <li key={c.id}>
            <span className={`case-badge-inline case-${c.status.replaceAll(" ", "-")}`}>{c.status}</span>{" "}
            {c.link ? <a href={c.link}>{c.title}</a> : c.title}: {c.number}
          </li>
        ))}
      </ul>
      <p><a className="button secondary" href="/cases">Every case, with what it can&apos;t do <ArrowRight size={16} aria-hidden="true" /></a></p>
    </section>
  );
}
