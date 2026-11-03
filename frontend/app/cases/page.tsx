"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { ProofNav } from "@/components/ProofNav";
import { StatusChip } from "@/components/StatusChip";

// One row per real-world case (scripts/build_cases.py): answer first (status, title, the problem in one line, the
// number against an independent truth), evidence when the row is opened (who, what they get, the models, the status
// rule and the ladder, the limits). The status is the highest rung of the ladder the case has evidence for.

interface Rung { rung: string; evidence: string }
interface Meaning { rung: string; meaning: string }
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
  const [ladder, setLadder] = useState<Meaning[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState("All");
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      // The strongest evidence leads: highest rung first.
      .then((d) => {
        setCases([...d.cases].sort((a: Case, b: Case) => LADDER.indexOf(a.status) - LADDER.indexOf(b.status)));
        setLadder(d.ladder ?? []);
      })
      .catch((e) => setError(`The cases could not be loaded (${(e as Error).message}).`));
  }, []);
  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of cases ?? []) m.set(c.status, (m.get(c.status) ?? 0) + 1);
    return LADDER.filter((r) => m.has(r)).map((r) => ({ rung: r, n: m.get(r)! }));
  }, [cases]);
  const shown = (cases ?? []).filter((c) => group === "All" || GROUP[c.id] === group);
  const inGroup = (g: string) => (cases ?? []).filter((c) => g === "All" || GROUP[c.id] === g).length;

  return (
    <main id="main">
      <ProofNav />
      <header className="page-head-split">
        <h1>Real-world cases</h1>
        <p className="tagline">
          Each case ends in something a person can act on, is tested on real data and validated against an independent
          answer. Nothing is trained: each case is fitted from scratch. The status comes from each result file by a written rule.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {cases && (
        <>
          <ul className="stat-strip count-strip" aria-label={`${cases.length} cases by status`}>
            {counts.map((c) => (
              <li key={c.rung}><span className="stat-number">{c.n}</span><span className="stat-label">{c.rung}</span></li>
            ))}
          </ul>
          <div className="chips" role="group" aria-label="Filter cases">
            {GROUPS.map((g) => (
              <button key={g} type="button" className={`chip${group === g ? " on" : ""}`} aria-pressed={group === g} onClick={() => setGroup(g)}>
                {g} {inGroup(g)}</button>
            ))}
          </div>
          <ul className="case-rows">
            {shown.map((c, i) => (
              <li key={c.id}>
                <details className="case-row" open={i === 0 && group === "All"}>
                  <summary>
                    <span className="case-row-status"><StatusChip status={c.status} /><span className="small">{GROUP[c.id]}</span></span>
                    <span className="case-row-main">
                      <strong className="case-row-title" id={`case-${c.id}`}>{c.title}</strong>
                      <span className="small">{c.problem}</span>
                      <span className="case-row-number">{c.number ?? "No result yet."}</span>
                    </span>
                    <ChevronDown size={20} className="case-row-chev" aria-hidden="true" />
                  </summary>
                  <div className="case-row-body">
                    <dl className="facts">
                      <dt>Who</dt><dd>{c.who}</dd>
                      <dt>What they get</dt><dd>{c.gives}</dd>
                      {c.models && <><dt>Meta models</dt><dd>{c.models}</dd></>}
                      <dt>Status rule</dt><dd>{c.status_rule}
                        {c.ladder && c.ladder.length > 1 && (
                          <ul className="small">{c.ladder.map((r) => <li key={r.rung}><strong>{label(r.rung)}:</strong> {r.evidence}</li>)}</ul>
                        )}</dd>
                      <dt>What it can&apos;t do</dt><dd>{c.cant}</dd>
                    </dl>
                    {c.link && <a className="button" href={c.link}>See the result</a>}
                  </div>
                </details>
              </li>
            ))}
          </ul>
          <section className="stack" style={{ gap: 12 }} aria-labelledby="ladder-title">
            <h2 id="ladder-title">How a status is earned</h2>
            {ladder.length > 0 && (
              <dl className="ladder-list">
                {ladder.map((r) => <div key={r.rung}><dt><StatusChip status={r.rung} /></dt><dd>{r.meaning}</dd></div>)}
              </dl>
            )}
            <p className="small">The ladder, lowest to highest: not yet validated, partial, proven, predicted (a blind prediction
              registered before the answer existed), confirmed (reality scored it a hit), improves a public tool, solved (someone outside
              the project used it; recorded, never inferred). Open predictions are on the <a href="/predictions">predictions page</a>.</p>
          </section>
        </>
      )}
    </main>
  );
}
