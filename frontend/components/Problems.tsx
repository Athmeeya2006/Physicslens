"use client";

import { ArrowRight, Camera, ChevronRight, Orbit, ScanSearch, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { StatusChip } from "./StatusChip";

// The home page's cases ledger: one row per case, its status and its number against an independent answer, then the
// four ways to try it yourself. Read from the same
// generated file as /cases (scripts/build_cases.py), whose status rules are written down, so this page can never claim
// more than the results do. The highest rung of the status ladder comes first.

interface Case { id: string; title: string; problem: string; who: string; number: string | null; status: string; link: string | null;
  usage?: { who: string; date?: string }[] }

// The status ladder, highest first (scripts/build_cases.py): evidence beyond the check itself ranks above "proven".
const LADDER = ["solved", "improves a public tool", "confirmed", "predicted", "proven", "partial", "not yet validated"];
const ORDER: Record<string, number> = Object.fromEntries(LADDER.map((r, i) => [r, i]));
const STRIP = 99;

const TRY = [
  { href: "#analyze-title", icon: ScanSearch, title: "Analyze a recording",
    text: "Upload a clip: the equation, its parameters with intervals, and the law's ghost on your video." },
  { href: "/live", icon: Camera, title: "Live camera", text: "Point your phone at something moving and watch the law converge as you film." },
  { href: "/audit", icon: ShieldCheck, title: "Physics audit", text: "Check a video for motion no physical law allows, when there is enough to judge." },
  { href: "/sky", icon: Orbit, title: "Solar System Lab", text: "Change the real Solar System, then see which law the agent finds from the new motion." },
];

export function TryIt() {
  return (
    <section className="band" aria-labelledby="try-title">
      <div className="band-inner stack" style={{ gap: 20 }}>
        <h2 id="try-title">Try it. Everything is open, no code needed.</h2>
        <ul className="try-tiles">
          {TRY.map((t) => (
            <li key={t.title}>
              <a className="tile" href={t.href}>
                <t.icon size={22} strokeWidth={1.8} aria-hidden="true" />
                <strong>{t.title}</strong>
                <span className="small">{t.text}</span>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

export function Problems() {
  const [cases, setCases] = useState<Case[]>([]);
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : null)).then((d) => d && setCases(d.cases)).catch(() => setCases([]));
  }, []);
  const shown = cases.filter((c) => c.number).sort((a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9));
  const proven = shown.filter((c) => ORDER[c.status] <= ORDER.proven).length;
  return (
    <section className="band band-surface" aria-labelledby="problems-title">
      <div className="band-inner stack" style={{ gap: 16 }}>
        <h2 id="problems-title">Problems it solves, each checked against an independent answer.</h2>
        {shown.length > 0 && (
          <>
            <p>
              {proven} of {shown.length} cases proven or higher (highest status reached: {shown[0].status}). Nothing is
              trained: each case is fitted from scratch.
            </p>
            <ul className="ledger">
              {shown.slice(0, STRIP).map((c) => {
                const row = (
                  <>
                    <StatusChip status={c.status} />
                    <strong className="ledger-title">{c.title}</strong>
                    <span className="ledger-number">{c.number}</span>
                    {c.link ? <ChevronRight size={18} className="ledger-chev" aria-hidden="true" /> : <span />}
                  </>
                );
                return <li key={c.id}>{c.link ? <a className="ledger-row" href={c.link}>{row}</a> : <div className="ledger-row">{row}</div>}</li>;
              })}
            </ul>
            <p className="small">
              <strong>Used by:</strong>{" "}
              {shown.flatMap((c) => c.usage ?? []).length
                ? shown.flatMap((c) => (c.usage ?? []).map((u) => `${u.who} (${c.title})`)).join("; ")
                : "open for outside use. Every result, its data and its exports are public, and each recorded use will be listed here."}
            </p>
            <p><a href="/cases">All {shown.length} cases, and how each was checked against an independent answer <ArrowRight size={14} aria-hidden="true" /></a></p>
          </>
        )}
      </div>
    </section>
  );
}
