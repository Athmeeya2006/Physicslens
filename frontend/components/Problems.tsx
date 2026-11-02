"use client";

import { ArrowRight, Bot, Globe2, Ruler } from "lucide-react";
import { useEffect, useState } from "react";

// The home page after the hero: three ways in (robotics, the real-world cases, measuring your own clip), then a compact
// strip of the cases, each one line of its number against an independent answer and its status. Read from the same
// generated file as /cases (scripts/build_cases.py), whose status rules are written down, so this page can never claim
// more than the results do. The highest rung of the status ladder comes first.

interface Case { id: string; title: string; problem: string; who: string; number: string | null; status: string; link: string | null;
  usage?: { who: string; date?: string }[] }

// The status ladder, highest first (scripts/build_cases.py): evidence beyond the check itself ranks above "proven".
const LADDER = ["solved", "improves a public tool", "confirmed", "predicted", "proven", "partial", "not yet validated"];
const ORDER: Record<string, number> = Object.fromEntries(LADDER.map((r, i) => [r, i]));
const STRIP = 99;

const TILES = [
  { href: "/?run=gallery-paperbag", icon: Bot, title: "Robotics sim-to-real",
    text: "A thrown object's drag law from one video, exported to MuJoCo and checked on flight it never saw." },
  { href: "/cases", icon: Globe2, title: "Real-world cases",
    text: "Meteorite falls, exoplanet timing, the Sun's rotation: each answer checked against an independent one." },
  { href: "#analyze-title", icon: Ruler, title: "Measure anything",
    text: "Film something moving: the equation, its parameters with intervals, and a ghost of the law on the video." },
];

export function Problems() {
  const [cases, setCases] = useState<Case[]>([]);
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : null)).then((d) => d && setCases(d.cases)).catch(() => setCases([]));
  }, []);
  const shown = cases.filter((c) => c.number).sort((a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9));
  const proven = shown.filter((c) => ORDER[c.status] <= ORDER.proven).length;
  return (
    <section className="stack" style={{ gap: 14 }} aria-labelledby="problems-title">
      <h2 id="problems-title" className="visually-hidden">Ways in</h2>
      <ul className="tiles">
        {TILES.map((t) => (
          <li key={t.title}>
            <a className="tile" href={t.href}>
              <t.icon size={22} strokeWidth={1.8} aria-hidden="true" />
              <strong>{t.title}</strong>
              <span className="small muted">{t.text}</span>
            </a>
          </li>
        ))}
      </ul>
      {shown.length > 0 && (
        <div className="panel stack cases-strip" style={{ gap: 8 }}>
          <p className="small">
            <strong>Checked against independent answers:</strong> {proven} of {shown.length} cases proven or higher (highest status
            reached: {shown[0].status}). Nothing is trained: each case is fitted from scratch.
          </p>
          <ul>
            {shown.slice(0, STRIP).map((c) => (
              <li key={c.id} className="strip-row">
                <span className={`case-badge-inline case-${c.status.replaceAll(" ", "-")}`}>{c.status}</span>
                {c.link ? <a href={c.link}><strong>{c.title}</strong></a> : <strong>{c.title}</strong>}
                <span className="strip-number small muted">{c.number}</span>
              </li>
            ))}
          </ul>
          <p className="small">
            <strong>Used by:</strong>{" "}
            {shown.flatMap((c) => c.usage ?? []).length
              ? shown.flatMap((c) => (c.usage ?? []).map((u) => `${u.who} (${c.title})`)).join("; ")
              : "nobody outside the project yet. Only recorded use is listed here, so this stays empty until there is some."}
          </p>
          <p className="small"><a href="/cases">All {shown.length} cases, how each was checked, and what it can&apos;t do <ArrowRight size={14} aria-hidden="true" /></a></p>
        </div>
      )}
    </section>
  );
}
