"use client";

import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { ProofNav } from "@/components/ProofNav";

// One row per real-world case (scripts/build_cases.py): answer first (status, title, the problem in one line, the
// number against an independent truth), evidence when the row is opened (who, what they get, the models, the status
// rule and the ladder, the limits). The status is the highest rung of the ladder the case has evidence for.

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

export default function CasesPage() {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState("All");
  useEffect(() => {
    fetch("/cases.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      // The strongest evidence leads: highest rung first.
      .then((d) => {
        setCases([...d.cases].sort((a: Case, b: Case) => LADDER.indexOf(a.status) - LADDER.indexOf(b.status)));
      })
      .catch((e) => setError(`The cases could not be loaded (${(e as Error).message}).`));
  }, []);
  const shown = (cases ?? []).filter((c) => group === "All" || GROUP[c.id] === group);
  const inGroup = (g: string) => (cases ?? []).filter((c) => g === "All" || GROUP[c.id] === g).length;

  return (
    <main id="main">
      <ProofNav />
      <header className="page-head-split">
        <h1>Real-world cases</h1>
        <p className="tagline">
          Each case ends in something a person can act on, is tested on real data and validated against an independent
          answer. Nothing is trained: each case is fitted from scratch.
        </p>
      </header>
      {error && <p className="error">{error}</p>}
      {cases && (
        <>
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
                    <span className="case-row-status"><span className="small">{GROUP[c.id]}</span></span>
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
                      <dt>Scope</dt><dd>{c.cant}</dd>
                    </dl>
                    {c.link && <a className="button" href={c.link}>See the result</a>}
                  </div>
                </details>
              </li>
            ))}
          </ul>
          <p className="small">Forecasts registered before the answer exists are on the <a href="/predictions">predictions page</a>.</p>
        </>
      )}
    </main>
  );
}
